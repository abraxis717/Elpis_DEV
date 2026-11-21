"""Compact immutable-by-interface geodesic identity index; no transport authority."""
from array import array
from bisect import bisect_left
from hashlib import sha256
from struct import pack
import Hexaflag as hxf

DEPTH=7
CAPACITY=7**DEPTH
SHIFT=18
MASK=(1<<SHIFT)-1
DIGIT={d:i for i,d in enumerate(hxf.DIGITS)}

def code_of(address,depth=None):
    if type(address) is not tuple or not address or type(address[0]) is not int:
        raise ValueError('ADDRESS_TYPE')
    d=address[0]
    if not 0<=d<=DEPTH or len(address)!=d+1 or (depth is not None and d!=depth):
        raise ValueError('ADDRESS_DEPTH')
    n=0
    for digit in address[1:]:
        if type(digit) is not tuple or len(digit)!=2 or any(type(x) is not int for x in digit) or digit not in DIGIT:
            raise ValueError('ADDRESS_DIGIT')
        n=7*n+DIGIT[digit]
    return n

def address_of(code,depth=DEPTH):
    if type(depth) is not int or not 0<=depth<=DEPTH or type(code) is not int or not 0<=code<7**depth:
        raise ValueError('CODE_RANGE')
    digits=[None]*depth
    for i in range(depth-1,-1,-1):
        code,d=divmod(code,7); digits[i]=hxf.DIGITS[d]
    return (depth,*digits)

class GeoIndex:
    __slots__=('_codes','_inverse','_sorted','_edges','_neighbors','_degrees','_lattice')
    def __init__(self,codes,edges):
        n=len(codes)
        if not 12<=n<=144002 or len(edges)!=3*n-6:
            raise ValueError('GEOMETRY_CAPACITY')
        self._codes=array('I',codes)
        self._inverse=array('i',[-1])*CAPACITY
        for v,c in enumerate(self._codes):
            if not 0<=c<CAPACITY or self._inverse[c]!=-1:
                raise ValueError('DUPLICATE_OR_INVALID_CODE')
            self._inverse[c]=v
        self._sorted=array('I',sorted(self._codes))
        self._edges=array('Q',edges)
        self._neighbors=array('i',[-1])*(6*n)
        self._degrees=bytearray(n)
        previous=-1
        for e in self._edges:
            u,v=e>>SHIFT,e&MASK
            if not previous<e or not 0<=u<v<n or self._degrees[u]>=6 or self._degrees[v]>=6:
                raise ValueError('EDGE_INVALID')
            previous=e
            for a,b in ((u,v),(v,u)):
                self._neighbors[6*a+self._degrees[a]]=b
                self._degrees[a]+=1
        if self._degrees.count(5)!=12 or self._degrees.count(6)!=n-12:
            raise ValueError('DEGREE_INVALID')
        self._lattice=hxf.HexaflagLattice(max_depth=DEPTH)
    @property
    def vertices(self): return len(self._codes)
    @property
    def edges(self): return len(self._edges)
    def _vertex(self,v):
        if type(v) is not int or not 0<=v<self.vertices: raise ValueError('VERTEX_RANGE')
    def vertex_code(self,v):
        self._vertex(v); return self._codes[v]
    def address(self,v): return address_of(self.vertex_code(v))
    def resolve_code(self,c):
        if type(c) is not int or not 0<=c<CAPACITY: raise ValueError('CODE_RANGE')
        v=self._inverse[c]
        if v<0: raise KeyError('UNOCCUPIED_ADDRESS')
        return v
    def resolve(self,address): return self.resolve_code(code_of(address,DEPTH))
    def coordinate(self,v): return self._lattice.decode(self.address(v))
    def resolve_coordinate(self,xy): return self.resolve(self._lattice.encode(xy,DEPTH))
    def _span(self,prefix):
        p=code_of(prefix); f=7**(DEPTH-prefix[0])
        return bisect_left(self._sorted,p*f),bisect_left(self._sorted,(p+1)*f)
    def prefix_count(self,prefix):
        lo,hi=self._span(prefix); return hi-lo
    def prefix_page(self,prefix,offset=0,limit=1024):
        lo,hi=self._span(prefix)
        if type(offset) is not int or not 0<=offset<=hi-lo or type(limit) is not int or not 1<=limit<=1024:
            raise ValueError('PAGE_RANGE')
        return tuple(self._inverse[c] for c in self._sorted[lo+offset:min(hi,lo+offset+limit)])
    def neighbors(self,v):
        self._vertex(v); return tuple(self._neighbors[6*v:6*v+self._degrees[v]])
    def arc_id(self,u,v):
        self._vertex(u); self._vertex(v)
        if u==v: raise KeyError('NONEDGE')
        e=(min(u,v)<<SHIFT)|max(u,v)
        i=bisect_left(self._edges,e)
        if i==self.edges or self._edges[i]!=e: raise KeyError('NONEDGE')
        return 2*i+int(u>v)
    def arc_endpoints(self,r):
        if type(r) is not int or not 0<=r<2*self.edges: raise ValueError('ARC_RANGE')
        e=self._edges[r//2]; u,v=e>>SHIFT,e&MASK
        return (v,u) if r%2 else (u,v)
    def payload_bytes(self):
        return sum(len(a)*a.itemsize for a in (self._codes,self._inverse,self._sorted,self._edges,self._neighbors))+len(self._degrees)
    def mapping_sha256(self):
        h=sha256()
        for v,c in enumerate(self._codes): h.update(pack('<II',v,c))
        return h.hexdigest()
    def edge_sha256(self):
        h=sha256()
        for e in self._edges: h.update(pack('<II',e>>SHIFT,e&MASK))
        return h.hexdigest()
