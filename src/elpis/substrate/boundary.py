"""Fail-closed Linux descriptor capabilities for file-backed resources.

A trusted root is opened once; every later open resolves beneath the retained
descriptor with ``openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)``. There is no
pathname-walk emulation: kernels/platforms without openat2 are unsupported and
fail closed. Native libraries are copied into a sealed memfd, hashed against an
independent pin, and only then loaded, so the bytes that were verified are the
bytes that run. Other Elpis modules remain importable on every platform.
"""
import ctypes as C
import errno
import os
from pathlib import Path
import platform
import stat
import sys
from threading import Lock

from .contracts import Code, ContractError, require
from .digests import raw_sha256


class _OpenHow(C.Structure):
    _fields_ = [('flags', C.c_uint64), ('mode', C.c_uint64), ('resolve', C.c_uint64)]


def _openat2(dirfd, path, flags):
    # Audited Linux syscall numbers; unknown ABIs must not guess a syscall.
    require(sys.platform == 'linux' and platform.machine() in
            ('x86_64', 'aarch64', 'riscv64') and C.sizeof(C.c_void_p) == 8,
            Code.UNSUPPORTED, 'openat2 platform/ABI')
    libc = C.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = C.c_long
    how = _OpenHow(flags, 0, 0x08 | 0x04)  # BENEATH | NO_SYMLINKS (includes magic links)
    fd = syscall(C.c_long(437), C.c_int(dirfd), C.c_char_p(os.fsencode(path)),
                 C.byref(how), C.c_size_t(C.sizeof(how)))
    if fd < 0:
        error = C.get_errno()
        code = (Code.UNSUPPORTED if error in (errno.ENOSYS, errno.EINVAL, errno.EPERM)
                else Code.MISSING if error == errno.ENOENT else Code.IO)
        # EAGAIN (rename race), EXDEV, ELOOP, etc. all fail closed; never downgrade.
        raise ContractError(code, 'openat2 resolution: ' + os.strerror(error))
    return fd


class RootCapability:
    """Root selected once; all subsequent opens are relative to the retained FD.

    The root's identity is a trusted deployment choice. Renaming it later does
    not redirect this capability. Mount administration and kernel are trusted.
    """
    def __init__(self, root):
        self.path = Path(root)
        self.fd = -1
        require(self.path.is_absolute() and '..' not in self.path.parts and '\x00' not in str(self.path),
                detail='absolute trusted root')
        require(sys.platform == 'linux', Code.UNSUPPORTED, 'Linux secure file admission required')
        anchor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            self.fd = _openat2(anchor, str(self.path).lstrip('/') or '.',
                               os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        finally:
            os.close(anchor)

    def relative(self, path):
        path = Path(path)
        require('..' not in path.parts, detail='path traversal')
        if path.is_absolute():
            require(path.is_relative_to(self.path), detail='path outside root')
            path = path.relative_to(self.path)
        require(str(path) != '.' and '\x00' not in str(path), detail='asset relative path')
        return str(path)

    def open_file(self, path):
        require(self.fd >= 0, Code.CLOSED, 'root capability')
        # NONBLOCK avoids blocking on an attacker-supplied FIFO before fstat.
        fd = _openat2(self.fd, self.relative(path),
                      os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode), Code.INTEGRITY, 'regular file required')
            return fd
        except BaseException:
            os.close(fd)
            raise

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def stamp(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def read_exact(fd, size, offset):
    data = bytearray()
    while len(data) < size:
        try:
            part = os.pread(fd, size - len(data), offset + len(data))
        except InterruptedError:
            continue
        require(bool(part), Code.IO, 'short descriptor read')
        data.extend(part)
    return bytes(data)


# ctypes does not automatically dlclose. Retain each successful load's FD for
# process lifetime so dlopen cannot reuse a cached /proc/self/fd/N name for a
# different library. Closing the FD with the provider would create that hazard.
_LOADED_NATIVE_OBJECTS = {}
_NATIVE_LOAD_LOCK = Lock()


def load_native(root, path, expected):
    """Copy opened bytes, seal, hash the sealed object, then dlopen its FD.

    Pin covers the top-level ELF only. Loader/dependencies/environment are trusted.
    No fallback to the candidate pathname, or to an unsealed inode, is permitted.
    """
    require(sys.platform == 'linux' and hasattr(os, 'memfd_create'),
            Code.UNSUPPORTED, 'sealed native loading')
    import fcntl
    source = root.open_file(path)
    sealed = -1
    loaded = False
    try:
        before = os.fstat(source)
        require(before.st_size == expected.size, Code.IDENTITY, 'native size')
        sealed = os.memfd_create('elpis-verified-provider', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        offset = 0
        while offset < expected.size:
            data = read_exact(source, min(1024 * 1024, expected.size - offset), offset)
            view = memoryview(data)
            while view:
                try:
                    written = os.write(sealed, view)
                except InterruptedError:
                    continue
                require(written > 0, Code.IO, 'native snapshot write')
                view = view[written:]
            offset += len(data)
        require(stamp(before) == stamp(os.fstat(source)), Code.INTEGRITY, 'native changed during copy')
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(sealed, fcntl.F_ADD_SEALS, seals)
        require(fcntl.fcntl(sealed, fcntl.F_GET_SEALS) & seals == seals,
                Code.UNSUPPORTED, 'native seals unavailable')
        digest = raw_sha256()
        for offset in range(0, expected.size, 1024 * 1024):
            digest.update(read_exact(sealed, min(1024 * 1024, expected.size - offset), offset))
        require(digest.hexdigest() == expected.sha256, Code.IDENTITY, 'native SHA-256')
        # Repeated contexts share code but never mutable FMS state. Validate
        # every new candidate above, even when its executable identity is cached.
        key = (expected.size, expected.sha256)
        with _NATIVE_LOAD_LOCK:
            if key in _LOADED_NATIVE_OBJECTS:
                return _LOADED_NATIVE_OBJECTS[key][1]
            lib = C.CDLL(f'/proc/self/fd/{sealed}', mode=os.RTLD_NOW | os.RTLD_LOCAL)
            _LOADED_NATIVE_OBJECTS[key] = (sealed, lib)
            loaded = True
            return lib
    except OSError as exc:
        code = Code.UNSUPPORTED if exc.errno in (errno.ENOSYS, errno.EPERM, errno.EINVAL) else Code.IO
        raise ContractError(code, 'sealed native load') from exc
    finally:
        os.close(source)
        if sealed >= 0 and not loaded:
            os.close(sealed)
