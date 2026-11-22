"""Geodome reference API v1.0.0. Trusted single-thread consumer interface."""
from array import array
from GEO144K import g1c_compact as geo, g2b_bridge as bridge
from geodome_index import GeoIndex, address_of, code_of, CAPACITY, DEPTH
from geodome_state import NodeState, StateHandle, MAX_WIDTH, MAX_BATCH
from geodome_scheduler import SparseScheduler, Round, MAX_ENROLLED, MAX_ARCS, MAX_ROUNDS, MAX_VISITS
API_VERSION='1.0.0'
SOURCE_PIN='993429cc94e9e3dcfd42df40502fa5b57175d883'
MAPPING_SHA256='abe96202b554608bd8f5e702b70459daaaf4ab33eb7dfd058723fd4d1724f51b'
EDGE_SHA256='712a8fc3d94685aeb614456ba8c88f6df7e6c6ea8a4c19e86d4d3865ad014231'
__all__=('API_VERSION','SOURCE_PIN','MAPPING_SHA256','EDGE_SHA256','build_index','new_state','GeoIndex','NodeState','StateHandle','SparseScheduler','Round','address_of','code_of','CAPACITY','DEPTH','MAX_WIDTH','MAX_BATCH','MAX_ENROLLED','MAX_ARCS','MAX_ROUNDS','MAX_VISITS')

def build_index():
    """Construct the fixed frequency-120 index; reject any identity drift."""
    xyz,faces=geo.build(120); n=len(xyz)//3
    edges=bridge.collect_edges(faces,n); del xyz,faces
    bfs,_=bridge.graph_bfs_order(edges,n); codes=array('I',[0])*n
    for rank,v in enumerate(bfs): codes[v]=bridge.rank_to_code(rank,n)
    idx=GeoIndex(codes,edges)
    if (idx.vertices,idx.edges,idx.mapping_sha256(),idx.edge_sha256())!=(144002,432000,MAPPING_SHA256,EDGE_SHA256):
        raise RuntimeError('V1_IDENTITY_DRIFT')
    return idx

def new_state(index,*,width=16,precision='f64',mode='sparse',capacity=None):
    if type(index) is not GeoIndex: raise TypeError('INDEX_TYPE')
    if capacity is None: capacity=index.vertices if mode=='dense' else min(4096,index.vertices)
    return NodeState(index,width=width,precision=precision,mode=mode,capacity=capacity)
