"""Bounded sorted/deduplicated row lookup over HACF FMS file assets.

Row I/O and E4M3/E8M0 + BF16 rounding mechanics reference
ElpisDonors/Qwen3.8/ds4_nvme/ds4_engram.c, revision
0aaea5a238fb41a35106a551e73c8409dfb751ac (MIT; retained license in
LICENSES/DS4-MIT.txt).
Elpis errors replace donor process-aborting/thread-error behavior.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from .contracts import Bank, Code, ContractError, RowIdentity, identity, integer, require


def row_representation(asset,offset,rows,dimension,codec):
    return identity('row-representation',dict(asset=asset,offset=offset,rows=rows,
                    dimension=dimension,codec=codec,packing='row-major-v1'))


def row_bytes(dimension,codec):
    integer(dimension,1)
    if codec=='F32_LE': return dimension*4
    if codec=='DS4_E4M3_E8M0_BF16':
        require(dimension%32==0,detail='FP8 scale blocks')
        return dimension+dimension//32
    raise ContractError(Code.UNSUPPORTED,'row codec')


def decode_row(raw,dimension,codec):
    require(len(raw)==row_bytes(dimension,codec),Code.ENCODING,'row length')
    if codec=='F32_LE':
        result=np.frombuffer(raw,dtype='<f4').copy()
    else:
        codes=np.frombuffer(raw[:dimension],dtype=np.uint8)
        scales=np.frombuffer(raw[dimension:],dtype=np.uint8)
        require(not np.any((codes&127)==127) and not np.any(scales==255),Code.ENCODING,'invalid FP8 code')
        exponent=((codes>>3)&15).astype(np.int32)
        mantissa=(codes&7).astype(np.float32)
        base=np.where(exponent==0,np.ldexp(mantissa,-9),np.ldexp(mantissa+8,exponent-10))
        with np.errstate(over='ignore',invalid='ignore'):
            values=np.ldexp(base,np.repeat(scales.astype(np.int32)-127,32))
            values=np.where(codes&128,-values,values).astype('<f4')
        require(np.all(np.isfinite(values)),Code.ENCODING,'FP8 overflow')
        bits=values.view('<u4')
        bits=(bits+np.uint32(0x7fff)+((bits>>16)&1)) & np.uint32(0xffff0000)
        result=bits.view('<f4')
    require(np.all(np.isfinite(result)),Code.ENCODING,'non-finite row')
    return result


@dataclass(frozen=True)
class RowTable:
    bank: Bank
    asset: str
    offset: int
    codec: str

    def validate(self,provider,expected_bank):
        require(self.bank.digest==expected_bank,Code.IDENTITY,'bank pin')
        integer(self.offset)
        require(self.asset in provider._assets,Code.MISSING,'row asset')
        manifest=provider._assets[self.asset][1]
        stride=row_bytes(self.bank.dimension,self.codec)
        require(self.offset+self.bank.rows*stride<=manifest.size,detail='table range')
        require(self.bank.content==manifest.content and self.bank.representation==
                row_representation(self.asset,self.offset,self.bank.rows,self.bank.dimension,self.codec),
                Code.IDENTITY,'row content/representation')


class RowEngine:
    def __init__(self,provider,table,*,expected_bank,workers=1,max_rows=4096,max_output_bytes=16<<20):
        table.validate(provider,expected_bank)
        integer(workers,1,32); integer(max_rows,1); integer(max_output_bytes,1)
        self.provider=provider; self.table=table; self.workers=workers
        self.max_rows=max_rows; self.max_output_bytes=max_output_bytes
        self.last_metrics={}

    def lookup(self,requests):
        require(type(requests) is tuple and len(requests)<=self.max_rows,Code.LIMIT,'row batch')
        bank=self.table.bank; stride=row_bytes(bank.dimension,self.table.codec)
        require(len(requests)*bank.dimension*4<=self.max_output_bytes,Code.LIMIT,'caller output budget')
        positions={}
        for i,r in enumerate(requests):
            require(type(r) is RowIdentity and r.bank==bank.digest,Code.IDENTITY,'row bank')
            require(r.row<bank.rows,detail='row out of range')
            positions.setdefault(r.row,[]).append(i)
        output=np.empty((len(requests),bank.dimension),dtype='<f4')
        before=self.provider.stats(); start=perf_counter_ns()
        def read(row):
            with self.provider.acquire(self.table.asset,self.table.offset+row*stride,stride) as lease:
                raw=lease.read()
            return row,decode_row(raw,bank.dimension,self.table.codec)
        def consume(results):
            for row,vector in results:
                for index in positions[row]: output[index]=vector
        ordered=sorted(positions)
        try:
            if self.workers==1: consume(map(read,ordered))
            else:
                # Submit only one bounded wave; no unbounded future/decoded-row queue.
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for start_row in range(0,len(ordered),self.workers):
                        consume(pool.map(read,ordered[start_row:start_row+self.workers]))
        except (RuntimeError,MemoryError,OSError) as exc:
            raise ContractError(Code.IO,'row worker/resource failure') from exc
        after=self.provider.stats(); physical=after['pread_bytes']-before['pread_bytes']
        semantic=len(requests)*stride
        self.last_metrics=dict(row_requests=len(requests),unique_rows=len(ordered),semantic_bytes=semantic,
                               pread_bytes=physical,read_amplification=physical/semantic if semantic else 0,
                               lookup_ns=perf_counter_ns()-start,workers=self.workers,
                               row_staging_upper_bound=self.workers*(stride+bank.dimension*32),
                               row_staging_upper_bound_kind='analytical_bound',
                               cache_hits=after['hits']-before['hits'],cache_misses=after['misses']-before['misses'])
        return output
