#!/usr/bin/env python3
"""Bounded-RAM Class-I spherical triangulation and overlapping star audit.

Pure stdlib, no imports from CGC/YM and no persistent writes.
Full-frequency (120) mode is explicitly gated by the shell runner, not invoked by default.
"""
from __future__ import annotations
from array import array
from collections import Counter
from hashlib import sha256
from math import sqrt, isfinite
from struct import pack
import argparse, json, sys

PHI=(1.0+sqrt(5.0))/2.0
RAW=((-1,PHI,0),(1,PHI,0),(-1,-PHI,0),(1,-PHI,0),
     (0,-1,PHI),(0,1,PHI),(0,0-1,-PHI),(0,1,-PHI),
     (PHI,0,-1),(PHI,0,1),(-PHI,0,-1),(-PHI,0,1))
BASE=tuple(tuple(float(x)/sqrt(sum(y*y for y in p)) for x in p) for p in RAW)
FACES=((0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
       (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
       (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
       (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1))
BASE_EDGES=tuple(sorted({tuple(sorted(pair)) for a,b,c in FACES for pair in ((a,b),(b,c),(c,a))}))
EDGE_ID={e:i for i,e in enumerate(BASE_EDGES)}
assert len(BASE_EDGES)==30
MAX_VERTICES=144_002
MASK18=(1<<18)-1
MASK19=(1<<19)-1


def vertex_id(f, fid, a,b,c,i,j,k):
    w={a:i,b:j,c:k}
    positive=[p for p,v in w.items() if v]
    if len(positive)==1: return positive[0]
    if len(positive)==2:
        lo,hi=sorted(positive)
        return 12+EDGE_ID[lo,hi]*(f-1)+(w[hi]-1)
    assert len(positive)==3 and i>0 and j>0 and k>0
    face_interior=(f-1)*(f-2)//2
    idx=(i-1)*(f-1)-((i-1)*i)//2+(j-1)
    assert 0<=idx<face_interior
    return 12+30*(f-1)+fid*face_interior+idx


def build(f):
    if not isinstance(f,int) or not 1<=f<=120:
        raise ValueError('FREQUENCY_OUT_OF_SCOPE_1_TO_120')
    V=10*f*f+2
    assert V<=MAX_VERTICES
    coords=array('d',[0.0])*(3*V)
    faces=array('I')
    seen=bytearray(V)
    for fid,(a,b,c) in enumerate(FACES):
        local=array('I',[0])*((f+1)*(f+2)//2)
        for i in range(f+1):
            row=i*(f+1)-i*(i-1)//2
            for j in range(f-i+1):
                k=f-i-j
                vid=vertex_id(f,fid,a,b,c,i,j,k)
                assert vid<V
                x=(i*BASE[a][0]+j*BASE[b][0]+k*BASE[c][0])/f
                y=(i*BASE[a][1]+j*BASE[b][1]+k*BASE[c][1])/f
                z=(i*BASE[a][2]+j*BASE[b][2]+k*BASE[c][2])/f
                n=sqrt(x*x+y*y+z*z)
                xyz=(x/n,y/n,z/n)
                if seen[vid]:
                    if max(abs(coords[3*vid+t]-xyz[t]) for t in range(3))>1e-12:
                        raise AssertionError('SEAM_COORDINATE_CONFLICT')
                else:
                    seen[vid]=1
                    coords[3*vid:3*vid+3]=array('d',xyz)
                local[row+j]=vid
        for i in range(f):
            r0=i*(f+1)-i*(i-1)//2
            r1=(i+1)*(f+1)-i*(i+1)//2
            for j in range(f-i):
                a0=local[r0+j];b0=local[r1+j];c0=local[r0+j+1]
                faces.extend((a0,b0,c0))
                if i+j<f-1:
                    d0=local[r1+j+1]
                    faces.extend((b0,d0,c0))
    if sum(seen)!=V:
        raise AssertionError('UNSEEN_VERTICES')
    if len(faces)!=60*f*f:
        raise AssertionError('FACE_COUNT_WRONG')
    return coords,faces


def audit(coords,faces,f,*,check_vertex_links=True):
    V=len(coords)//3; F=len(faces)//3; Eexpected=30*f*f
    if len(coords)!=3*V or V!=10*f*f+2 or F!=20*f*f:
        raise AssertionError('GEOMETRY_COUNTS')
    seen_faces=array('Q')
    records=array('Q')
    face_hash=sha256()
    degree_faces=array('H',[0])*V
    min_orientation=float('inf')
    for vid in range(V):
        x,y,z=coords[3*vid:3*vid+3]
        if not all(isfinite(v) for v in (x,y,z)) or abs(x*x+y*y+z*z-1)>1e-12:
            raise AssertionError('INVALID_VERTEX_COORDINATE')
    for fid in range(F):
        a,b,c=faces[3*fid:3*fid+3]
        if a>=V or b>=V or c>=V or len({a,b,c})!=3:
            raise AssertionError('INVALID_FACE')
        sa,sb,sc=sorted((a,b,c))
        seen_faces.append(sa | (sb<<18) | (sc<<36))
        face_hash.update(pack('<III',a,b,c))
        degree_faces[a]+=1;degree_faces[b]+=1;degree_faces[c]+=1
        ax,ay,az=coords[3*a:3*a+3]
        bx,by,bz=coords[3*b:3*b+3]
        cx,cy,cz=coords[3*c:3*c+3]
        abx,aby,abz=bx-ax,by-ay,bz-az
        acx,acy,acz=cx-ax,cy-ay,cz-az
        ox=aby*acz-abz*acy
        oy=abz*acx-abx*acz
        oz=abx*acy-aby*acx
        o=ox*(ax+bx+cx)+oy*(ay+by+cy)+oz*(az+bz+cz)
        if o<=1e-12: raise AssertionError('INVALID_FACE_ORIENTATION')
        min_orientation=min(min_orientation,o)
        for u,v in ((a,b),(b,c),(c,a)):
            lo,hi=(u,v) if u<v else (v,u)
            direction=0 if u<v else 1
            records.append((lo<<38)|(hi<<20)|(direction<<19)|fid)
    # Duplicate triangles are invalid even if face counts happen to match.
    sf=sorted(seen_faces)
    if any(sf[i]==sf[i-1] for i in range(1,len(sf))):
        raise AssertionError('DUPLICATE_TRIANGLE')
    del sf,seen_faces
    if any(d>6 for d in degree_faces):
        raise AssertionError('INVALID_STAR_DEGREE')
    sr=sorted(records)
    if len(sr)!=3*F or len(sr)!=2*Eexpected:
        raise AssertionError('EDGE_INCIDENCE_COUNT')
    offsets=array('I',[0])*(V+1)
    for v,n in enumerate(degree_faces):
        offsets[v+1]=offsets[v]+n
    if offsets[-1]!=3*F:
        raise AssertionError('TRIPLE_INCIDENCE')
    cursor=array('I',offsets[:-1]); star_ids=array('I',[0])*(3*F)
    for fid in range(F):
        for v in faces[3*fid:3*fid+3]:
            star_ids[cursor[v]]=fid
            cursor[v]+=1
    del cursor
    neighbors=array('i',[-1])*(6*V)
    degree=bytearray(V)
    edge_hash=sha256()
    for p in range(0,len(sr),2):
        x,y=sr[p],sr[p+1]
        if x>>20 != y>>20 or (x>>19)&1 != 0 or (y>>19)&1 != 1:
            raise AssertionError('NONMANIFOLD_OR_WRONG_EDGE_ORIENTATION')
        u=x>>38;v=(x>>20)&MASK18;f1=x&MASK19;f2=y&MASK19
        if f1==f2:raise AssertionError('EDGE_IN_SAME_FACE_TWICE')
        if degree[u]>=6 or degree[v]>=6:raise AssertionError('EXCESS_NEIGHBOR_DEGREE')
        neighbors[6*u+degree[u]]=v; degree[u]+=1
        neighbors[6*v+degree[v]]=u; degree[v]+=1
        # Exact two-triangle overlap follows from the edge's two incident faces.
        if not (f1 in star_ids[offsets[u]:offsets[u+1]] and
                f1 in star_ids[offsets[v]:offsets[v+1]] and
                f2 in star_ids[offsets[u]:offsets[u+1]] and
                f2 in star_ids[offsets[v]:offsets[v+1]]):
            raise AssertionError('EDGE_STAR_INCIDENCE_CONFLICT')
        edge_hash.update(pack('<II',u,v))
    del sr,records
    degcount=Counter(degree)
    if degcount!={5:12,6:V-12} and not (V==12 and degcount=={5:12}):
        raise AssertionError(f'SPHERICAL_DEFECTS_INVALID {degcount}')
    if any(degree_faces[v]!=degree[v] for v in range(V)):
        raise AssertionError('VERTEX_STAR_DEGREE_MISMATCH')
    passed_alternating=0; five_count=0
    if check_vertex_links:
        for v in range(V):
            adj={u:set() for u in neighbors[6*v:6*v+degree[v]]}
            vf=star_ids[offsets[v]:offsets[v+1]]
            for fid in vf:
                a,b,c=faces[3*fid:3*fid+3]
                other=[u for u in (a,b,c) if u!=v]
                if len(other)!=2 or other[0] not in adj or other[1] not in adj:
                    raise AssertionError('BROKEN_VERTEX_LINK')
                u,w=other
                adj[u].add(w);adj[w].add(u)
            if any(len(s)!=2 for s in adj.values()):
                raise AssertionError('VERTEX_LINK_NOT_CYCLE')
            anchor=min(adj)
            visited=[anchor];previous=None;current=anchor
            while True:
                choices=sorted(adj[current]-({previous} if previous is not None else set()))
                if not choices:raise AssertionError('BROKEN_RING')
                next_v=choices[0]
                if next_v==anchor:break
                if next_v in visited:raise AssertionError('DISCONNECTED_RING')
                visited.append(next_v)
                previous,current=current,next_v
            if len(visited)!=degree[v]:
                raise AssertionError('DISCONNECTED_VERTEX_LINK')
            if len(visited)==6:
                overlaps=[]
                for u in visited[::2]:
                    a=set(vf)
                    b=set(star_ids[offsets[u]:offsets[u+1]])
                    intersection=a&b
                    if len(intersection)!=2:raise AssertionError('TWO_TRIANGLE_OVERLAP_FAILED')
                    overlaps.append(intersection)
                if len(set.union(*overlaps))!=6 or sum(map(len,overlaps))!=6:
                    raise AssertionError('THREE_ALTERNATING_OVERLAPS_FAILED')
                passed_alternating+=1
            else: five_count+=1
    if V-30*f*f+F!=2:raise AssertionError('NOT_SPHERE')
    return dict(frequency=f,vertices=V,edges=Eexpected,faces=F,euler=2,
                valence5=degcount.get(5,0),valence6=degcount.get(6,0),
                two_triangle_overlaps=Eexpected,triangle_star_memberships=3*F,
                three_alternate_neighbor_cover=passed_alternating,
                five_valent_exception_count=five_count,
                min_outward_orientation=round(min_orientation,12),
                face_order_sha256=face_hash.hexdigest(),
                canonical_edges_sha256=edge_hash.hexdigest())


def reference_parity(f,coords,faces,g1_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_g1_geometry',g1_path)
    if spec is None or spec.loader is None:raise AssertionError('G1_NOT_IMPORTABLE')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    oldpos,oldfaces=module.build(f)
    def key(x):return tuple(round(z,11) for z in x)
    v_to_new={key(coords[i*3:i*3+3]):i for i in range(len(coords)//3)}
    if len(v_to_new)!=len(oldpos):raise AssertionError('REFERENCE_VERTEX_KEY_COLLISION')
    mapped=[]
    for o in oldpos:
        q=key(o)
        if q not in v_to_new:raise AssertionError('REFERENCE_COORDINATE_MISMATCH')
        mapped.append(v_to_new[q])
    expected=sorted(tuple(sorted(mapped[v] for v in old)) for old in oldfaces)
    actual=sorted(tuple(sorted(faces[3*i:3*i+3])) for i in range(len(faces)//3))
    if expected!=actual:raise AssertionError('REFERENCE_FACE_SET_MISMATCH')
    return True


def negative_controls(f):
    xyz,tri=build(f)
    flipped=array('I',tri)
    flipped[1],flipped[2]=flipped[2],flipped[1]
    try:
        audit(xyz,flipped,f)
    except AssertionError as exc:
        if str(exc)!='INVALID_FACE_ORIENTATION':raise AssertionError(f'WRONG_NEGATIVE_REJECTION {exc}')
    else:raise AssertionError('INVERTED_FACE_ACCEPTED')
    dup=array('I',tri)
    dup[0:3]=dup[3:6]
    try: audit(xyz,dup,f)
    except AssertionError as exc:
        if str(exc)!='DUPLICATE_TRIANGLE':raise AssertionError(f'WRONG_NEGATIVE_REJECTION {exc}')
    else:raise AssertionError('DUPLICATE_FACE_ACCEPTED')
    badxyz=array('d',xyz);badxyz[0]=999.0
    try:audit(badxyz,tri,f)
    except AssertionError as exc:
        if str(exc)!='INVALID_VERTEX_COORDINATE':raise AssertionError(f'WRONG_NEGATIVE_REJECTION {exc}')
    else:raise AssertionError('INVALID_VERTEX_ACCEPTED')
    return {'reversed_face_rejected':True,'duplicate_face_rejected':True,'invalid_coordinate_rejected':True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frequencies',type=int,nargs='+',default=[7,16,32])
    ap.add_argument('--g1-source',default='')
    ap.add_argument('--allow-full-144k',action='store_true')
    ap.add_argument('--skip-vertex-links',action='store_true')
    args=ap.parse_args()
    if args.skip_vertex_links:raise SystemExit('BLOCKED=EXHAUSTIVE_CHECK_REQUIRED')
    if any(not(1<=f<=120) for f in args.frequencies):raise SystemExit('BLOCKED=FREQUENCY_RANGE_1_120')
    if any(f>48 for f in args.frequencies) and not args.allow_full_144k:
        raise SystemExit('BLOCKED=EXPLICIT_FULL_SCALE_AUTHORIZATION_REQUIRED')
    print('PHASE=GEO144K_G1C_COMPACT_PRIMAL_DUAL_R0',flush=True)
    print('MODE=STANDARD_LIBRARY_ONLY_IN_MEMORY_NO_GPU_NO_CGC_NO_YM',flush=True)
    for f in args.frequencies:
        print('START_FREQUENCY='+str(f),flush=True)
        xyz,tri=build(f)
        result=audit(xyz,tri,f)
        if args.g1_source and f<=16:
            result['reference_g1_face_parity']=reference_parity(f,xyz,tri,args.g1_source)
        print('PASS_FREQUENCY='+str(f)+' '+json.dumps(result,sort_keys=True),flush=True)
        del xyz,tri
    print('NEGATIVE_CONTROLS='+json.dumps(negative_controls(2),sort_keys=True),flush=True)
    print('DISPOSITION=PASS_EXHAUSTIVE_GEODESIC_GEOMETRY_FOR_REQUESTED_FREQUENCIES',flush=True)
    print('CGC_YM_HCA88=NOT_EXECUTED',flush=True)
    print('PERSISTENT_RUNTIME_WRITES=NONE',flush=True)

if __name__=='__main__':main()
