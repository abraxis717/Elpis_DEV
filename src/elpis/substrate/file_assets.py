"""Read-only file-backed range assets over FMS ABI v2.

Buffered pread mode: native FMS RAM is bounded, Linux page cache is NOT charged
or bounded by this extension. Physical bytes mean bytes returned by pread,
not device-level I/O (which cannot be inferred from buffered reads).
Every page is verified before native registration or exposure to inference.
"""
from collections import OrderedDict
import ctypes as C
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from threading import RLock
from time import perf_counter_ns
from .authority import PinnedAuthority
from .boundary import RootCapability, load_native, read_exact, stamp as _stamp
from .contracts import Code, ContractError, digest_value, integer, require
from .digests import identity, raw_digest, raw_sha256


def bounded_path(root, path):
    """Legacy synthetic-fixture convenience; NOT a race-resistant boundary."""
    root, path = Path(root), Path(path)
    require(root.is_absolute() and '..' not in root.parts, detail='workspace root')
    path = path if path.is_absolute() else root / path
    require('..' not in path.parts and path.is_relative_to(root), Code.INVALID, 'path boundary')
    # Refuse symlinks without resolving their targets or traversing outside root.
    current = root
    require(not current.is_symlink(), detail='root symlink')
    for part in path.relative_to(root).parts:
        current = current / part
        require(not current.is_symlink(), detail='symlink forbidden')
    return path


@dataclass(frozen=True)
class AssetManifest:
    size: int
    page_size: int
    content: str
    pages: tuple[str, ...]
    schema: str = 'elpis.fms.file-asset.r0'

    def __post_init__(self):
        require(self.schema == 'elpis.fms.file-asset.r0')
        integer(self.size, 1)
        integer(self.page_size, 1, 16 * 1024 * 1024)
        digest_value(self.content)
        require(type(self.pages) is tuple and len(self.pages) == (self.size + self.page_size-1)//self.page_size,
                detail='page map geometry')
        for digest in self.pages:
            digest_value(digest)

    @property
    def digest(self):
        return identity('file-asset.manifest', self)


def _inspect_fd(fd, page_size, *, expected_size=None):
    """Observe one opened object. Returns metadata and ordinary raw SHA-256."""
    integer(page_size, 1, 16 * 1024 * 1024)
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode), detail='regular asset required')
    require(expected_size is None or before.st_size == expected_size,
            Code.INTEGRITY, 'asset geometry')
    pages, digest = [], raw_sha256()
    for offset in range(0, before.st_size, page_size):
        data = read_exact(fd, min(page_size, before.st_size-offset), offset)
        digest.update(data)
        pages.append(raw_digest(data))
    require(_stamp(before) == _stamp(os.fstat(fd)), Code.INTEGRITY, 'asset changed during intake')
    pages = tuple(pages)
    content = identity('file-asset.content-map',
                       dict(size=before.st_size, page_size=page_size, pages=pages))
    return AssetManifest(before.st_size, page_size, content, pages), digest.hexdigest(), _stamp(before)


def inspect_asset(root, path, page_size):
    """Untrusted observation only; never an independent authority or admission."""
    try:
        with RootCapability(root) as boundary:
            fd = boundary.open_file(path)
            try:
                return _inspect_fd(fd, page_size)[0]
            finally:
                os.close(fd)
    except OSError as exc:
        raise ContractError(Code.IO, 'asset intake') from exc


class _NativePages:
    def __init__(self, library, warm_bytes, max_pages, absent_policy):
        self.lib = library
        lib = self.lib
        vp, u64 = C.c_void_p, C.c_uint64
        signatures = {
            'elpis_fms_file_create_memory': ([u64,C.c_uint32,C.c_int,C.POINTER(vp)],C.c_int),
            'fms_register': ([vp,C.c_uint32,u64,C.c_int,C.c_float,vp,C.POINTER(u64)],C.c_int),
            'fms_unregister': ([vp,u64],C.c_int),
            'fms_lease_acquire': ([vp,u64,C.c_int,C.c_uint,C.POINTER(vp)],C.c_int),
            'fms_lease_ptr': ([vp],vp), 'fms_lease_tier': ([vp],C.c_int),
            'fms_lease_release': ([vp,vp],C.c_int),
            'elpis_fms_file_stats': ([vp,C.POINTER(u64)],C.c_int),
            'fms_destroy': ([vp],None),
        }
        for name,(args,result) in signatures.items():
            try:
                f=getattr(lib,name)
            except AttributeError as exc:
                raise ContractError(Code.UNSUPPORTED, 'native file-provider ABI') from exc
            f.argtypes=args; f.restype=result
        self.ctx=vp()
        self.check(lib.elpis_fms_file_create_memory(warm_bytes,max_pages,absent_policy,C.byref(self.ctx)))

    @staticmethod
    def check(rc):
        if rc < 0:
            code={-2:Code.LIMIT,-3:Code.MISSING,-4:Code.BUSY,-5:Code.UNSUPPORTED,
                  -6:Code.IO,-7:Code.LIMIT,-9:Code.INTEGRITY,-10:Code.DEVICE}.get(rc,Code.INVALID)
            raise ContractError(code,f'native FMS status {rc}')
        return rc

    def register(self,data):
        oid=C.c_uint64()
        buffer=C.create_string_buffer(data)
        self.check(self.lib.fms_register(self.ctx,0x50414745,len(data),1,0,buffer,C.byref(oid)))
        return oid.value

    def acquire(self,oid,tier):
        lease=C.c_void_p()
        self.check(self.lib.fms_lease_acquire(self.ctx,oid,tier,1,C.byref(lease)))
        return lease,self.lib.fms_lease_ptr(lease),self.lib.fms_lease_tier(lease)

    def release(self,lease):
        self.check(self.lib.fms_lease_release(self.ctx,lease))

    def evict(self,oid):
        self.check(self.lib.fms_unregister(self.ctx,oid))

    def stats(self):
        values=(C.c_uint64*8)()
        self.check(self.lib.elpis_fms_file_stats(self.ctx,values))
        return dict(zip(('hot','warm','native_cold','ram','device','native_storage','pages','pinned'),values))

    def close(self):
        self.lib.fms_destroy(self.ctx)


class RangeLease:
    def __init__(self,owner,parts,offset,length,actual_tier):
        self._owner=owner; self._parts=parts
        self.offset=offset; self.length=length; self.actual_tier=actual_tier
        self._active=True

    def read(self):
        with self._owner._lock:
            require(self._active,Code.CLOSED,'released range lease')
            return b''.join(C.string_at(ptr+start,size) for _,_,ptr,start,size in self._parts)

    def readinto(self,buffer):
        """Copy the leased range into a caller-owned writable buffer of exactly `length` bytes.

        One copy from the verified native pages, with no intermediate bytes objects, so a
        caller can reuse bounded scratch instead of rebuilding a buffer per range.
        """
        with self._owner._lock:
            require(self._active,Code.CLOSED,'released range lease')
            view=memoryview(buffer)
            require(view.contiguous and not view.readonly and view.nbytes==self.length,detail='readinto buffer')
            view=view.cast('B')
            target=(C.c_char*self.length).from_buffer(view)
            try:
                base=C.addressof(target); pos=0
                for _,_,ptr,start,size in self._parts:
                    C.memmove(base+pos,ptr+start,size); pos+=size
            finally:
                del target
                view.release()
            return self.length

    def bind_fence(self,fence):
        # CPU POSIX PAL has no device/fence facility. Never pretend completion.
        raise ContractError(Code.UNSUPPORTED,'CPU file-asset provider has no accelerator fences')

    def release(self):
        with self._owner._lock:
            require(self._active,Code.CLOSED,'double range release')
            for key,lease,_,_,_ in self._parts:
                self._owner._native.release(lease)
                self._owner._pins[key]-=1
            self._active=False

    def __enter__(self): return self
    def __exit__(self,*exc): self.release()


class FMSFileAssets:
    """Additive external COLD catalog + bounded native FMS page materialization.

    No writable replicas. Admission streams and verifies all bytes once. Eviction unregisters
    verified native pages; externally owned immutable storage is never deleted.
    A serialized I/O critical section prioritizes clear accounting in R0.
    """
    def __init__(self,*,root,library,authority=None,library_id=None,
                 warm_bytes=65536,staging_bytes=65536,
                 storage_bytes=1<<40,max_pages=1024,hot_absent_policy='FOLD_DOWN'):
        self._check_authority(authority)
        require(type(library_id) is str and library_id in authority.libraries, Code.IDENTITY, 'authorized native identifier required')
        self.authority = authority
        self.root=Path(root)
        integer(warm_bytes,1); integer(staging_bytes,1); integer(storage_bytes,1); integer(max_pages,1,(1<<32)-1)
        require(hot_absent_policy in ('FOLD_DOWN','REJECT'))
        # The RAM-only native provider has no writable COLD tier: it never
        # creates, traverses or opens any scratch location.
        self._boundary = RootCapability(root)
        try:
            lib = load_native(self._boundary, library, authority.libraries[library_id])
            self._native=_NativePages(lib,warm_bytes,max_pages,int(hot_absent_policy=='REJECT'))
        except BaseException:
            self._boundary.close()
            raise
        self.warm_budget=warm_bytes; self.staging_budget=staging_bytes; self.storage_budget=storage_bytes
        self._lock=RLock(); self._assets={}; self._pages=OrderedDict(); self._pins={}; self._closed=False
        self.telemetry={'semantic_bytes':0,'pread_bytes':0,'reads':0,'hits':0,'misses':0,
                        'read_ns':0,'integrity_ns':0,'staging_ns':0,'staging_high_water':0,
                        'staging_high_water_kind':'analytical_bound'}

    def _open(self): require(not self._closed,Code.CLOSED,'FMS file provider')

    @staticmethod
    def _check_authority(authority):
        require(type(authority) is PinnedAuthority and authority.provenance == 'deployment',
                Code.IDENTITY, 'independently pinned deployment authority required')

    def _authorized_identity(self, asset_id, path, manifest):
        require(type(asset_id) is str and asset_id in self.authority.assets, Code.IDENTITY, 'authorized asset identifier required')
        return self.authority.assets[asset_id]

    def register(self,path,manifest,*,asset_id=None,expected_manifest=None):
        with self._lock:
            self._open()
            require(type(manifest) is AssetManifest, Code.IDENTITY, 'asset observation required')
            trusted = self._authorized_identity(asset_id, path, manifest)
            require(manifest.digest == trusted.manifest_digest and
                    manifest.size == trusted.size and manifest.page_size == trusted.page_size,
                    Code.IDENTITY, 'independent asset identity')
            require(expected_manifest is None or manifest.digest == expected_manifest,
                    Code.IDENTITY, 'page-map root')
            require(manifest.page_size <= min(self.staging_budget//4,self.warm_budget),Code.LIMIT,'page staging (four-copy upper bound)')
            if manifest.digest not in self._assets:
                require(sum(v[1].size for v in self._assets.values())+manifest.size<=self.storage_budget,Code.LIMIT,'external COLD budget')
            fd = self._boundary.open_file(path)
            try:
                observed, sha256, stamp = _inspect_fd(fd, trusted.page_size, expected_size=trusted.size)
                require(sha256 == trusted.sha256 and observed == manifest,
                        Code.INTEGRITY, 'asset content against independent authority')
                # Even duplicate registration must validate the presented path.
                if manifest.digest in self._assets:
                    os.close(fd)
                    return manifest.digest
            except OSError as exc:
                os.close(fd)
                raise ContractError(Code.IO, 'asset admission read') from exc
            except BaseException:
                os.close(fd)
                raise
            self._assets[manifest.digest]=(fd,manifest,stamp)
            return manifest.digest

    def _load(self,asset,page):
        fd,m,stamp=self._assets[asset]
        require(_stamp(os.fstat(fd))==stamp,Code.INTEGRITY,'changed backing file')
        key=(asset,page)
        if key in self._pages:
            self.telemetry['hits']+=1
            self._pages.move_to_end(key)
            return self._pages[key]
        self.telemetry['misses']+=1
        start=page*m.page_size; count=min(m.page_size,m.size-start)
        data=bytearray()
        begin=perf_counter_ns()
        try:
            while len(data)<count:
                try: piece=os.pread(fd,count-len(data),start+len(data))
                except InterruptedError: continue
                require(bool(piece),Code.IO,'short backing read')
                self.telemetry['pread_bytes']+=len(piece)
                self.telemetry['reads']+=1
                data.extend(piece)
        except OSError as exc:
            raise ContractError(Code.IO,'pread') from exc
        self.telemetry['read_ns']+=perf_counter_ns()-begin
        begin=perf_counter_ns()
        require(raw_digest(data)==m.pages[page],Code.INTEGRITY,'backing page digest')
        require(_stamp(os.fstat(fd))==stamp,Code.INTEGRITY,'asset changed during read')
        self.telemetry['integrity_ns']+=perf_counter_ns()-begin
        self.telemetry['staging_high_water']=max(self.telemetry['staging_high_water'],4*count)
        begin=perf_counter_ns()
        while True:
            try:
                oid=self._native.register(bytes(data))
                break
            except ContractError as exc:
                if exc.code not in (Code.LIMIT,Code.UNSUPPORTED): raise
                victim=next((k for k in self._pages if self._pins[k]==0),None)
                require(victim is not None,Code.LIMIT,'all native pages leased or budget too small')
                self._native.evict(self._pages.pop(victim))
                del self._pins[victim]
        self.telemetry['staging_ns']+=perf_counter_ns()-begin
        self._pages[key]=oid; self._pins[key]=0
        return oid

    def acquire(self,asset,offset,length,*,tier='WARM'):
        with self._lock:
            self._open()
            require(asset in self._assets,Code.MISSING,'asset')
            integer(offset); integer(length,1)
            require(tier in ('HOT','WARM'),Code.UNSUPPORTED,'COLD is external storage, not an addressable lease')
            m=self._assets[asset][1]
            require(offset+length<=m.size,detail='range outside asset')
            first,last=offset//m.page_size,(offset+length-1)//m.page_size
            require((last-first+1)*m.page_size<=self.warm_budget,Code.LIMIT,'range exceeds native budget')
            self.telemetry['semantic_bytes']+=length
            parts=[]; actual_tiers=[]
            try:
                for page in range(first,last+1):
                    key=(asset,page); oid=self._load(asset,page)
                    lease,ptr,page_actual=self._native.acquire(oid,0 if tier=='HOT' else 1)
                    try:
                        require(page_actual in (0,1),Code.INTEGRITY,'native lease tier')
                    except BaseException:
                        self._native.release(lease)
                        raise
                    self._pins[key]+=1
                    start=max(offset,page*m.page_size)-page*m.page_size
                    end=min(offset+length,(page+1)*m.page_size)-page*m.page_size
                    parts.append((key,lease,ptr,start,end-start))
                    actual_tiers.append(page_actual)
            except BaseException:
                for key,lease,_,_,_ in parts:
                    self._native.release(lease); self._pins[key]-=1
                raise
            actual=max(actual_tiers)
            return RangeLease(self,parts,offset,length,('HOT','WARM')[actual])

    def manifest(self,asset):
        """Immutable manifest of an admitted asset (size, page size, page digests).

        Read-only inspection: admission already verified every byte against the
        independent authority; no range is read and no page is materialized.
        """
        with self._lock:
            self._open()
            require(type(asset) is str and asset in self._assets,Code.MISSING,'asset')
            return self._assets[asset][1]

    def evict(self,asset=None):
        with self._lock:
            self._open()
            keys=[key for key in self._pages if asset is None or key[0]==asset]
            require(not any(self._pins[key] for key in keys),Code.BUSY,'active range lease')
            for key in keys:
                self._native.evict(self._pages.pop(key)); del self._pins[key]

    def stats(self):
        with self._lock:
            self._open()
            return dict(self._native.stats(),external_cold=sum(v[1].size for v in self._assets.values()),
                        buffered_page_cache_accounted=False,**self.telemetry)

    def close(self):
        with self._lock:
            self._open()
            self.evict()
            for fd,_,_ in self._assets.values(): os.close(fd)
            self._assets.clear(); self._native.close(); self._boundary.close(); self._closed=True

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
