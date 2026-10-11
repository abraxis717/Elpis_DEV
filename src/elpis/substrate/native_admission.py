"""One native-artifact admission mechanism for every high-impact Elpis native library.

Elpis executes native code it has verified, never native code a pathname happens to name. Admission is the
existing substrate mechanism, generalized from the structure bridges to RuntimeCore, native K1, the K1 FMS adapter
and the continuity authority:

* **independent authority**: a deployment-pinned catalog (:class:`~elpis.substrate.authority.PinnedAuthority`)
  whose own SHA-256 arrives through trusted configuration and which lists each library's identifier, exact size
  and exact SHA-256. Inspecting a candidate never creates authority; a library id the catalog does not list is
  refused;
* **trusted descriptor root**: every library is opened beneath one root descriptor with
  ``openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`` (:class:`~elpis.substrate.boundary.RootCapability`): no path
  traversal, no symlink escape, regular files only;
* **sealed bytes**: the opened bytes are copied into a sealed memfd, hashed there and only then loaded from the
  memfd (:func:`~elpis.substrate.boundary.load_native`), so the verified bytes are the bytes that run;
* **verified dependencies**: a set is admitted in dependency order (each library may depend only on earlier ones).
  After loading, every dependency the dynamic loader binds by SONAME (``lib<library_id>.so``) is checked to be
  pinned bytes: one of the sealed objects, or a file whose mapped inode and SHA-256 equal the pin. A dependency the
  loader resolved anywhere else refuses the whole admission. (The structure bridges' generic dependencies, such as
  libc or SQLite, are trusted platform, as before.)

The ABI check stays with each binding (``RuntimeLibrary``, ``K1Library``, ...), which refuses a library whose ABI
version is not its own. Every admission is recorded: :func:`admission_of` tells whether a loaded library came
through admission, which the managed runtime requires of the K1 libraries it drives.

Non-claims: the catalog pin authenticates exact bytes, not a publisher; the loader, the kernel and the platform
libraries are trusted; nothing here protects against a malicious in-process caller.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import os
from threading import Lock

from .authority import NativeIdentity, PinnedAuthority
from .boundary import RootCapability, load_native, read_exact
from .contracts import Code, require
from .digests import raw_sha256

__all__ = ("AdmittedLibrary", "admission_of", "admit_libraries", "admit_library")

_RTLD_LAZY, _RTLD_NOLOAD, _RTLD_DI_LINKMAP = 0x1, 0x4, 2


@dataclass(frozen=True)
class AdmittedLibrary:
    """One library loaded from sealed, pinned bytes under an independent authority."""

    library_id: str
    identity: NativeIdentity
    authority_digest: str
    lib: object


_ADMITTED: dict[int, AdmittedLibrary] = {}   # id(loaded library) -> admission; libraries are never unloaded
_LOCK = Lock()


def admission_of(lib) -> AdmittedLibrary | None:
    """The admission of a loaded library, or None when it was loaded any other way."""
    found = _ADMITTED.get(id(lib))
    return found if found is not None and found.lib is lib else None


class _LinkMap(C.Structure):
    _fields_ = [("l_addr", C.c_void_p), ("l_name", C.c_char_p), ("l_ld", C.c_void_p), ("l_next", C.c_void_p),
                ("l_prev", C.c_void_p)]


def _resolved_name(soname: str) -> str | None:
    """The path of the object the dynamic loader binds ``soname`` to in this process (None: none loaded)."""
    libc = C.CDLL(None)
    libc.dlopen.restype, libc.dlopen.argtypes = C.c_void_p, [C.c_char_p, C.c_int]
    libc.dlinfo.restype, libc.dlinfo.argtypes = C.c_int, [C.c_void_p, C.c_int, C.c_void_p]
    libc.dlclose.restype, libc.dlclose.argtypes = C.c_int, [C.c_void_p]
    handle = libc.dlopen(soname.encode(), _RTLD_LAZY | _RTLD_NOLOAD)
    if not handle:
        return None
    try:
        link = C.POINTER(_LinkMap)()
        require(libc.dlinfo(handle, _RTLD_DI_LINKMAP, C.byref(link)) == 0 and bool(link), Code.IO,
                "dynamic loader link map")
        return os.fsdecode(link.contents.l_name or b"")
    finally:
        libc.dlclose(handle)


def _mapped_inode(path: str) -> tuple[int, int] | None:
    """(device, inode) of the file mapped at ``path`` in this process, from /proc/self/maps."""
    with open("/proc/self/maps", "rb") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and os.fsdecode(fields[5].rstrip(b"\n")) == path:
                major, minor = (int(v, 16) for v in fields[3].split(b":"))
                return os.makedev(major, minor), int(fields[4])
    return None


def _verify_bound(soname: str, admitted: AdmittedLibrary) -> None:
    """Refuse unless the object the loader binds ``soname`` to is exactly ``admitted``'s pinned bytes."""
    name = _resolved_name(soname)
    require(name is not None, Code.IDENTITY, f"dependency {soname} is not loaded")
    sealed = getattr(admitted.lib, "_name", None)
    if name == sealed:
        return
    # Another object (loaded earlier by a pathname) carries the SONAME: it is acceptable only if the inode mapped
    # into this process is the file we can read now and that file is the pinned bytes.
    require(os.path.isabs(name) and not name.startswith("/proc/self/fd/"), Code.IDENTITY,
            f"dependency {soname} is bound to an unverified object")
    mapped = _mapped_inode(name)
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        require(mapped == (st.st_dev, st.st_ino) and st.st_size == admitted.identity.size, Code.IDENTITY,
                f"dependency {soname} is bound to bytes other than its pin")
        digest = raw_sha256()
        for offset in range(0, st.st_size, 1 << 20):
            digest.update(read_exact(fd, min(1 << 20, st.st_size - offset), offset))
        require(digest.hexdigest() == admitted.identity.sha256, Code.IDENTITY,
                f"dependency {soname} is bound to bytes other than its pin")
    finally:
        os.close(fd)


def admit_libraries(root, authority: PinnedAuthority, entries) -> tuple[AdmittedLibrary, ...]:
    """Admit ``entries`` (``(library_id, path beneath root)`` pairs, in dependency order) from sealed, pinned bytes.

    Every id must be listed by the deployment authority; every file is opened beneath ``root`` without symlinks,
    sealed, hashed against its pin and loaded from the seal. Then every library of the set except the last, which
    later ones may depend on by SONAME ``lib<library_id>.so``, is checked to be what the loader binds that SONAME
    to. Any refusal admits nothing new (a library already sealed stays loaded, as sealed libraries are never
    unloaded).
    """
    require(type(authority) is PinnedAuthority, Code.IDENTITY, "native admission requires a PinnedAuthority")
    require(authority.provenance == "deployment", Code.IDENTITY, "native admission requires deployment provenance")
    require(type(entries) is tuple and bool(entries), Code.INVALID, "native admission entries")
    ids = [e[0] for e in entries if type(e) is tuple and len(e) == 2 and type(e[0]) is str]
    require(len(ids) == len(entries) and len(set(ids)) == len(ids), Code.INVALID, "distinct (library_id, path) pairs")
    for library_id in ids:
        require(library_id in authority.libraries, Code.IDENTITY, f"{library_id} is not in the deployment authority")
    admitted = []
    with _LOCK, RootCapability(root) as cap:
        for library_id, path in entries:
            identity = authority.libraries[library_id]
            lib = load_native(cap, path, identity)
            admitted.append(AdmittedLibrary(library_id, identity, authority.digest, lib))
        for dependency in admitted[:-1]:
            _verify_bound(f"lib{dependency.library_id}.so", dependency)
        for item in admitted:
            _ADMITTED[id(item.lib)] = item
    return tuple(admitted)


def admit_library(root, path, authority: PinnedAuthority, library_id: str) -> AdmittedLibrary:
    """Admit one library (no Elpis dependency verified beyond the platform; see :func:`admit_libraries`)."""
    return admit_libraries(root, authority, ((library_id, path),))[0]
