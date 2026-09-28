"""DSV4 compact target: the first inference driver behind :class:`elpis.inference.target.Target`.

A compact, synthetic-fixture model, not a production DeepSeek model and not
compatible with trained tables. Local sliding-window attention, shared
compressed KV, a sparse global index and SwiGLU experts follow the
architectural decomposition of DeepSeek V4.1 inference/model.py (MIT; see
LICENSES/PROVENANCE.md). Memory rows are addressed by the DSV4.1 engram
scheme. Arithmetic is explicit CPU F32; there is no GPU determinism claim.
No weights are shipped or downloaded: every tensor is supplied by the caller.
"""
from time import perf_counter_ns
import numpy as np
from ...associative import DSV41Parameters
from ...contracts import Code, RowIdentity, digest_value, identity, integer, require
from ...global_context import GlobalCandidate, IndexConfig, IndexMode, select_global, selected_values
from ...target import (
    LatentInput, NeuralState, StepReceipt, Tensor, numerical_profile, softmax, vector,
)


class CompactTarget:
    def __init__(self,config,weights,projections,scheme,rows,experts):
        self.config=config; self.weights=dict(weights); self.projections={p.channel:p for p in projections}
        d=config.dimension; v=config.vocab
        expected={'embedding':(v,d),'q':(d,d),'k':(d,d),'v':(d,d),'out':(d,v),
                  'router':(d,len(config.expert_ids)),'compress':(d,1)}
        require(set(weights)==set(expected),detail='model weight roles')
        for key,shape in expected.items(): require(type(weights[key]) is Tensor and weights[key].shape==shape,detail='model geometry')
        require(set(self.projections)=={'M','G','X','R'} and len(projections)==4,detail='latent channels')
        for p in projections:
            require(p.model==config.model and p.weights.shape[1]==d,Code.IDENTITY,'latent projection model/dimension')
        require(scheme.parameters.tokenizer==config.tokenizer,Code.IDENTITY,'target tokenizer')
        raw_vocab=len(scheme.parameters.token_map) if type(scheme.parameters) is DSV41Parameters else scheme.parameters.vocab
        require(raw_vocab==v,Code.IDENTITY,'target vocabulary')
        scheme.validate_bank(rows.table.bank)
        require(rows.table.bank.model==config.model and experts.model==config.model,Code.IDENTITY,'target assets model')
        require(self.projections['M'].weights.shape[0]==rows.table.bank.dimension,detail='memory projection dimension')
        for i in config.expert_ids+config.shared_experts:
            m=experts.manifest(config.layer,i)
            require(m.tensors[0].shape[0]==d,detail='expert target dimension')
        self.scheme=scheme; self.rows=rows; self.experts=experts
        self.numerical_profile=numerical_profile()
        self.model_identity=identity('target-model',dict(config=config.digest,
              weights=tuple((k,t.digest) for k,t in sorted(weights.items())),
              projections=tuple(p.digest for p in projections),parameters=scheme.parameters.digest,
              bank=rows.table.bank.digest,experts=experts.digest))
        self.index_config=IndexConfig(config.model,weights['k'].digest,d,config.global_top_k,4,4)
        self.last_metrics={}

    def initial(self,context_snapshot):
        digest_value(context_snapshot)
        return NeuralState(self.model_identity,context_snapshot,self.numerical_profile,(),self.scheme.initial())

    def _latent(self,packet,context):
        require(type(packet) is LatentInput and packet.channel in self.projections,detail='latent packet')
        projection=self.projections[packet.channel]
        digest_value(packet.source)
        require(packet.context_snapshot==context and packet.projection==projection.digest and
                packet.source_schema==projection.source_schema,Code.IDENTITY,'latent provenance/projection')
        require(type(packet.values) is tuple and len(packet.values)==projection.weights.shape[0] and
                all(np.isfinite(x) for x in packet.values),detail='latent input geometry')
        return np.asarray(packet.values,dtype='<f4')@projection.weights.array()

    def step(self,state,token,*,expected_state,latents=(),resident_experts=None,index_mode=IndexMode.FULL):
        start=perf_counter_ns(); c=self.config; w={k:t.array() for k,t in self.weights.items()}
        require(type(state) is NeuralState and state.digest==expected_state,Code.STALE,'target state')
        require(state.model==self.model_identity,Code.IDENTITY,'target state model')
        require(state.numerical_profile==self.numerical_profile,Code.UNSUPPORTED,'numerical execution profile')
        integer(token,0,c.vocab-1)
        require(len(state.tokens)<c.max_tokens,Code.LIMIT,'explicit target context capacity')
        require(state.history.position==len(state.tokens),Code.STALE,'token/hash position')
        require(type(latents) is tuple and len({p.channel for p in latents})==len(latents),detail='latent channels')
        begin=perf_counter_ns()
        hashed=self.scheme.hash(state.history,(token,),expected_history=state.history.digest)
        hash_ns=perf_counter_ns()-begin
        p=self.scheme.parameters
        layer_index=p.layers.index(self.rows.table.bank.layer) if type(p) is DSV41Parameters else 0
        row_ids=hashed.rows[0][layer_index]
        memory=self.rows.lookup(tuple(RowIdentity(self.rows.table.bank.digest,r) for r in row_ids)).mean(axis=0,dtype=np.float32)
        mp=self.projections['M']
        internal=LatentInput('M',mp.source_schema,hashed.digest,state.context_snapshot,mp.digest,vector(memory))
        require(all(p.channel!='M' for p in latents),detail='associative channel is target-owned')
        x=w['embedding'][token].copy()+self._latent(internal,state.context_snapshot)
        for packet in latents:
            value=self._latent(packet,state.context_snapshot)
            # R is a recorded proposal only; mathematical routing is learned.
            if packet.channel!='R': x+=value
        begin=perf_counter_ns()
        key=x@w['k']; val=x@w['v']; query=x@w['q']
        keys=(state.local_keys+(vector(key),))[-c.local_window:]
        values=(state.local_values+(vector(val),))[-c.local_window:]
        score=np.asarray(keys,dtype='<f4')@query/np.float32(c.dimension**.5)
        local=softmax(score)@np.asarray(values,dtype='<f4')
        local_ns=perf_counter_ns()-begin
        pending=state.pending+((vector(key),vector(val),float((x@w['compress'])[0])),)
        pool=state.global_pool
        if len(pending)==c.compression:
            a=softmax([item[2] for item in pending])
            ck=a@np.asarray([item[0] for item in pending],dtype='<f4')
            cv=a@np.asarray([item[1] for item in pending],dtype='<f4')
            source=identity('native-kv-source',dict(model=state.model,tokens=state.tokens+(token,),context=state.context_snapshot))
            pool+=(GlobalCandidate(str(len(pool)),c.model,self.index_config.projection,len(state.tokens),vector(ck),vector(cv),source),)
            pending=()
        begin=perf_counter_ns()
        previous=None if index_mode is IndexMode.FULL else state.index
        index=select_global(self.index_config,pool,vector(query),context_snapshot=state.context_snapshot,
                            position=len(state.tokens),mode=index_mode,previous=previous,
                            expected_previous=None if previous is None else previous.digest)
        selected=selected_values(pool,index)
        global_value=np.zeros(c.dimension,dtype='<f4')
        if selected:
            score=np.asarray([v.key for v in selected],dtype='<f4')@query/np.float32(c.dimension**.5)
            global_value=softmax(score)@np.asarray([v.value for v in selected],dtype='<f4')
        global_ns=perf_counter_ns()-begin
        hidden=np.tanh(x+local+global_value).astype('<f4')
        route_logits=hidden@w['router']
        indexes=sorted(range(len(c.expert_ids)),key=lambda i:(-float(route_logits[i]),c.expert_ids[i]))[:c.active_experts]
        route=tuple(c.expert_ids[i] for i in indexes)
        route_weights=tuple(float(v) for v in softmax(route_logits[indexes]))
        hidden+=self.experts.execute(hidden,model=c.model,layer=c.layer,route=route,weights=route_weights,
                                    shared=c.shared_experts,resident=resident_experts)
        logits=hidden@w['out']
        require(np.all(np.isfinite(logits)),Code.ENCODING,'target logits')
        out=NeuralState(state.model,state.context_snapshot,state.numerical_profile,state.tokens+(token,),hashed.history,
                        keys,values,pending,pool,index,vector(hidden),vector(logits))
        expert_manifests=tuple(self.experts.manifest(c.layer,i) for i in route+c.shared_experts)
        assets=tuple(sorted({self.rows.table.asset}|{t.asset for m in expert_manifests for t in m.tensors}))
        receipt=StepReceipt(state.model,c.tokenizer,state.numerical_profile,state.digest,state.context_snapshot,token,p.schema,p.digest,
                            hashed.digest,self.rows.table.bank.digest,row_ids,index.digest,
                            tuple(packet.digest for packet in (internal,)+latents),route,
                            tuple(m.digest for m in expert_manifests),assets,out.digest)
        self.last_metrics=dict(hash_ns=hash_ns,local_context_ns=local_ns,global_index_ns=global_ns,
                               target_ns=perf_counter_ns()-start,rows=dict(self.rows.last_metrics),
                               experts=dict(self.experts.last_metrics),device_copy_ns=None)
        return out,receipt
