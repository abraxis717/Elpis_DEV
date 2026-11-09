#!/usr/bin/env python3
"""One graph genuine PyTorch tangent replay, independent HCA unbatched parity and gauge preflight."""
import json,torch as T
import r78,r64,r77,r62_source as r62,r63_source as r63,portable_mechanics as pm,r30_source as r30
import verify_r78 as V
T.set_num_threads(1)
def test():
 V.source_gate();sealed=V.report_of('YM_AWR_R78_RESULT.zip');V.validate(sealed)
 p=r78.source_gate();models=r78.models_from_r77(p)
 refs={(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']):r for r in p['rows']}
 graph='R77_OOD_D2';rows,native,eps,gauge=r78.get_data_for_graph(graph,models,refs)
 expected={(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']):r for r in sealed['rows']}
 dif=0.;checks=0
 for r in rows:
  key=(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']);rr=expected[key]
  for nm in r['tangents']:
   dif=max(dif,abs(rr['tangents'][nm]['log_gain']-r['tangents'][nm]['log_gain']))
  checks+=1
 r78.need(checks==72 and dif<2e-10,'SPOT_TANGENT_REPLAY')
 # One source-pinned original, single-model reference vs batched trace + gauge covariance.
 r61=pm.read_json('FROZEN_PROTOCOL_R61.json');adj=pm.graph_adjs(r77.CFG['graphs'][graph],r61)[1]
 m=models['54007/STATE8_TF1'];st=r64.pair_state(57703,'weak',160,batch=4)
 gs={j:pm.rotation(.13+.19*j,-.27-.1*j) for j in range(4)}
 with T.no_grad():
  base=r64.rollout(st,adj,m,20,keep=(8,12,20))
  rotated=r64.rollout(pm.gauge_state(st,gs),pm.gauged_links(adj,gs),m,20,keep=(8,12,20))
  cov=max(float((rotated[h][ch]-pm.gauge_state(base[h],gs)[ch]).abs().max()) for h in [8,12,20] for ch in [0,1])
 r78.need(cov<1e-9,'GAUGE_COVARIANCE')
 print(f'R78_FROZEN_DYNAMICS_TANGENT_SPOT={checks}/{checks} MAX_LOG_GAIN_DELTA={dif:.3g} GAUGE_MAX={cov:.3g} GENERATOR_IDENTITY_MAX={gauge:.3g}',flush=True)
if __name__=='__main__':test()
