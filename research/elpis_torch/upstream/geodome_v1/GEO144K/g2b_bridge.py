#!/usr/bin/env python3
# Modified for HexaFlake publication: historical imports rebased to public modules.
"""G2B: bounded deterministic spherical-vertex -> frozen Hexaflag address bridge.

The physical geometry remains the G1C icosahedron. The Hexaflag R1 integer
hierarchy is LOGICAL, not a physically nested hexagonal tiling. No YM relations
are registered and no model cores or checkpoints are imported.
"""
from __future__ import annotations
from array import array
from collections import deque
from hashlib import sha256
from struct import pack
import argparse
import json
import random
import sys

from . import g1c_compact as geo
import Hexaflag as hxf

DEPTH = 7
CAPACITY = 7 ** DEPTH
VERTEX_CAP = 144_002
SHIFT = 18
MASK = (1 << SHIFT) - 1


def require(ok, reason):
    if not ok:
        raise AssertionError(reason)


def collect_edges(faces, vertices):
    """One canonical record per oriented-triangle side; require 2 per edge."""
    raw = array('Q')
    for fid in range(len(faces)//3):
        a,b,c = faces[3*fid:3*fid+3]
        for u,v in ((a,b),(b,c),(c,a)):
            if u > v:
                u,v = v,u
            raw.append((u << SHIFT) | v)
    ordered = sorted(raw)
    del raw
    require(len(ordered) == 60 * ((vertices-2)//10), 'SIDE_COUNT_DRIFT')
    edges = array('Q')
    for i in range(0,len(ordered),2):
        require(i+1 < len(ordered) and ordered[i] == ordered[i+1],
                'EDGE_TWO_FACE_INCIDENCE_DRIFT')
        if i+2 < len(ordered):
            require(ordered[i] != ordered[i+2], 'EDGE_MORE_THAN_TWO_FACES')
        edges.append(ordered[i])
    return edges


def graph_bfs_order(edges, n):
    """Spherical graph ordering; only genuine triangle-mesh neighbor edges."""
    counts = bytearray(n)
    nbrs = array('I', [0]) * (6*n)
    for e in edges:
        u,v = (e >> SHIFT), (e & MASK)
        require(u<n and v<n and u<v, 'INVALID_EDGE_ID')
        require(counts[u] < 6 and counts[v] < 6, 'INVALID_NEIGHBOR_CAPACITY')
        nbrs[6*u+counts[u]] = v
        counts[u] += 1
        nbrs[6*v+counts[v]] = u
        counts[v] += 1
    require(sum(counts) == 2*len(edges), 'DEGREE_SUM_DRIFT')
    require(sum(c==5 for c in counts)==12, 'TWELVE_DEFECTS_NOT_FOUND')
    require(all(c in (5,6) for c in counts), 'INVALID_VERTEX_DEGREE')
    require(all(counts[v]==5 for v in range(12)), 'ICOSAHEDRAL_DEFECT_ANCHOR_DRIFT')
    visited=bytearray(n)
    queue=deque([0])
    visited[0]=1
    result=array('I')
    while queue:
        u=queue.popleft()
        result.append(u)
        for v in sorted(nbrs[6*u:6*u+counts[u]]):
            if not visited[v]:
                visited[v]=1
                queue.append(v)
    require(len(result)==n and all(visited), 'GEODESIC_GRAPH_DISCONNECTED')
    return result,counts


def rank_to_code(rank, n):
    """Balanced seven-way partition of a linear vertex order, at each depth.

    Each parent contains a contiguous interval in the source ordering. This
    locality is an EXTRA POLICY; R1's algebra does not itself provide it.
    """
    require(type(rank) is int and 0 <= rank < n <= CAPACITY, 'RANK_OUT_OF_RANGE')
    lo,hi=0,n
    code=0
    for _ in range(DEPTH):
        span=hi-lo
        q,r=divmod(span,7)
        rel=rank-lo
        large=(q+1)*r
        if rel < large:
            d=rel//(q+1)
        else:
            require(q>0, 'EMPTY_LEAF_SELECTION')
            d=r+(rel-large)//q
        require(0<=d<7, 'BAD_DIGIT')
        next_lo=lo+d*q+min(d,r)
        lo=next_lo
        hi=lo+q+int(d<r)
        code=7*code+d
    require(hi-lo==1 and lo==rank, 'RANK_LEAF_COLLISION')
    return code


def address_of(code):
    require(0<=code<CAPACITY, 'ADDRESS_CODE_RANGE')
    digs=[0]*DEPTH
    for i in range(DEPTH-1,-1,-1):
        code,digs[i]=divmod(code,7)
    return (DEPTH,*(hxf.DIGITS[d] for d in digs))


def lcp(a,b):
    z=0
    divisor=7**(DEPTH-1)
    for _ in range(DEPTH):
        if (a//divisor)%7 != (b//divisor)%7:
            break
        z+=1
        divisor//=7
    return z


def locality(edges,codes):
    acc=0
    root_shared=0
    level3_shared=0
    for e in edges:
        u,v=e>>SHIFT,e&MASK
        n=lcp(codes[u],codes[v])
        acc+=n
        root_shared+=n>=1
        level3_shared+=n>=3
    count=len(edges)
    return {
        'mean_shared_digits':round(acc/count,6),
        'same_root_child_edges':root_shared,
        'same_root_child_fraction':round(root_shared/count,6),
        'same_level3_parent_edges':level3_shared,
        'same_level3_parent_fraction':round(level3_shared/count,6)
    }


def verify_bijection(codes,n,edges,lattice):
    require(n<=CAPACITY, 'INSUFFICIENT_HEXAFLAG_CAPACITY')
    inverse=array('i',[-1])*CAPACITY
    counts=[0]*7
    for v,code in enumerate(codes):
        require(0<=code<CAPACITY, 'CODE_NOT_HEXAFLAG_REPRESENTABLE')
        require(inverse[code] == -1,'DUPLICATE_ADDRESS')
        inverse[code]=v
        counts[code//(7**(DEPTH-1))]+=1
    require(max(counts)-min(counts)<=1 and min(counts)>0,'UNBALANCED_ROOT_PARTITION')
    digest=sha256()
    for v,code in enumerate(codes):
        digest.update(pack('<II',v,code))
    for e in edges:
        u,v=e>>SHIFT,e&MASK
        require(inverse[codes[u]]==u and inverse[codes[v]]==v,
                'NEIGHBOR_IDENTITY_NOT_PRESERVED')
    # R1 is exercised against every address, rather than just treating a base-7
    # digit string as if it were an instance of R1's integer lattice.
    coordinate_owner={}
    for v,code in enumerate(codes):
        address=address_of(code)
        require(lattice.validate(address),'R1_REJECTED_MAPPED_ADDRESS')
        xy=lattice.decode(address)
        require(xy not in coordinate_owner,'R1_COORDINATE_COLLISION')
        coordinate_owner[xy]=v
        # Only a fixed sample requires the substantially slower seven-pass
        # exact coset division. The algebraic decode is exhaustive.
        if v < 32 or v >= n-32 or (v % max(1,n//512) == 0):
            require(lattice.encode(xy,DEPTH)==address,'R1_ENCODE_DECODE_MISMATCH')
            require(lattice.parent(address) == (DEPTH-1,*address[1:-1]),
                    'R1_PARENT_DRIFT')
            require(address in lattice.children(lattice.parent(address)),
                    'R1_CHILD_DRIFT')
    return digest.hexdigest(),counts,inverse,len(coordinate_owner)



def prefix_connectivity(codes, edges, n):
    """Count disconnected induced graph components within logical prefix groups.

    This is a qualification diagnostic, not a proof that a partition is
    geometrically convex, physically nested, or a spherical tiling.
    """
    result={}
    for depth in (1,2,3):
        divisor=7**(DEPTH-depth)
        parent=list(range(n))
        sizes=[1]*n
        def root(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]]
                x=parent[x]
            return x
        for e in edges:
            u,v=e>>SHIFT,e&MASK
            if codes[u]//divisor!=codes[v]//divisor:
                continue
            a,b=root(u),root(v)
            if a!=b:
                if sizes[a]<sizes[b]:a,b=b,a
                parent[b]=a
                sizes[a]+=sizes[b]
        components={}
        for v in range(n):
            key=codes[v]//divisor
            components.setdefault(key,set()).add(root(v))
        counts=tuple(len(c) for c in components.values())
        result[str(depth)]={
            'occupied_prefix_groups':len(counts),
            'disconnected_prefix_groups':sum(c>1 for c in counts),
            'worst_components_per_prefix':max(counts),
            'all_connected':all(c==1 for c in counts)
        }
    return result

def reject_controls(codes,inverse,edges,n,lattice):
    # Duplicate logical identity must be invalid even for geometrically
    # distinct vertices. Test the same guard as the normal registration path.
    def unique_map(seq):
        seen=set()
        for code in seq:
            require(code not in seen,'DUPLICATE_ADDRESS')
            seen.add(code)
    broken=array('I',codes)
    broken[1]=broken[0]
    try: unique_map(broken)
    except AssertionError as ex: require(str(ex)=='DUPLICATE_ADDRESS','WRONG_DUPLICATE_REJECTION')
    else:raise AssertionError('DUPLICATE_ADDRESS_ACCEPTED')
    a=address_of(codes[0]); p=lattice.parent(a)
    corrupt=(p[0],*p[1:-1],hxf.DIGITS[(hxf.DIGITS.index(p[-1])+1)%7])
    try: require(p==corrupt,'INVALID_ANCESTRY')
    except AssertionError as ex:require(str(ex)=='INVALID_ANCESTRY','WRONG_ANCESTRY_REJECTION')
    else:raise AssertionError('FALSE_ANCESTRY_ACCEPTED')
    e_set=set(edges)
    bad=None
    for v in range(n-1,0,-1):
        u=0;lo,hi=(u,v) if u<v else (v,u)
        packed=(lo<<SHIFT)|hi
        if packed not in e_set:
            bad=packed
            break
    require(bad is not None,'NEGATIVE_NONEDGE_NOT_FOUND')
    try:require(bad in e_set,'FABRICATED_NEIGHBOR')
    except AssertionError as ex:require(str(ex)=='FABRICATED_NEIGHBOR','WRONG_NEIGHBOR_REJECTION')
    else:raise AssertionError('FABRICATED_NEIGHBOR_ACCEPTED')
    return {'duplicate_address_rejected':True,
            'invalid_ancestry_rejected':True,
            'fabricated_neighbor_rejected':True}


def run(f):
    require(1<=f<=120,'FREQUENCY_NOT_ALLOWED')
    coords,faces=geo.build(f)
    a=geo.audit(coords,faces,f)
    n=a['vertices']
    require(7**6<n or f!=120, 'TARGET_DEPTH_NOT_REQUIRED')
    edges=collect_edges(faces,n)
    require(len(edges)==a['edges'],'EDGE_PARITY_MISMATCH')
    bfs,degree=graph_bfs_order(edges,n)
    codes_for_rank=array('I',(rank_to_code(i,n) for i in range(n)))
    bfs_codes=array('I',[0])*n
    for r,v in enumerate(bfs): bfs_codes[v]=codes_for_rank[r]
    index_codes=codes_for_rank
    rng=random.Random(20261119)
    shuffled=list(range(n));rng.shuffle(shuffled)
    random_codes=array('I',[0])*n
    for r,v in enumerate(shuffled):random_codes[v]=codes_for_rank[r]
    lat=hxf.HexaflagLattice(max_depth=DEPTH)
    identity_hash,root_groups,inverse,unique_coords=verify_bijection(bfs_codes,n,edges,lat)
    controls=reject_controls(bfs_codes,inverse,edges,n,lat)
    metrics={
        'frequency':f,'vertices':n,'edges':len(edges),'faces':a['faces'],
        'address_depth':DEPTH,'address_capacity':CAPACITY,
        'root_group_sizes':root_groups,
        'mapped_unique_r1_coordinates':unique_coords,
        'five_valent_defects_addressed':sum(degree[v]==5 for v in range(n)),
        'address_mapping_sha256':identity_hash,
        'bfs_graph_locality':locality(edges,bfs_codes),
        'bfs_prefix_connectivity':prefix_connectivity(bfs_codes,edges,n),
        'vertex_id_prefix_connectivity':prefix_connectivity(index_codes,edges,n),
        'vertex_id_order_baseline':locality(edges,index_codes),
        'random_order_baseline':locality(edges,random_codes),
        'negative_controls':controls,
        'ym_relations_registered':0,
        'physical_footprint_nesting_claim':False,
        'r3_registry_directly_qualified':False,
        'spatial_prefix_hierarchy_qualified':False,
    }
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frequencies', type=int,nargs='+',default=[7,16])
    ap.add_argument('--allow-full',action='store_true')
    args=ap.parse_args()
    if any(f>48 for f in args.frequencies) and not args.allow_full:
        raise SystemExit('BLOCKED=FULL_SCALE_EXPLICIT_GATE_REQUIRED')
    hxf.check_core()
    print('PHASE=GEO144K_G2B_FROZEN_HEXAFLAG_SPHERICAL_ADDRESS_BRIDGE_R0',flush=True)
    print('MODE=STD_LIB_IN_MEMORY_NO_GPU_NO_CGC_NO_YM_NO_WEIGHT_MUTATION',flush=True)
    print('HEX_ADDRESS_DEPTH=7 CAPACITY=823543 REQUIRED_VERTICES=144002',flush=True)
    for f in args.frequencies:
        print('START_FREQUENCY='+str(f),flush=True)
        r=run(f)
        print('PASS_FREQUENCY='+str(f)+' '+json.dumps(r,sort_keys=True),flush=True)
    print('DISPOSITION=PASS_LOGICAL_IDENTITY_HOLD_GEODESIC_PREFIX_CONNECTIVITY',flush=True)
    print('R3_REGISTRY=NOT_EXERCISED',flush=True)
    print('CGC_YM_HCA88=NOT_EXERCISED',flush=True)
    print('RUNTIME_PERSISTENT_WRITES=NONE',flush=True)

if __name__=='__main__':main()
