"""R36 CPU native-vs-donor parity science. No checkpoints, no GPU."""
import torch
from r35.slot_write import ERSASlotWriteRead
from r36.native_bridge import build_native,infer_native

def one_seed(seed,lib):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    m=ERSASlotWriteRead(d_model=8,num_slots=8,topk=2,max_steps=16)
    err_state=err_output=0.
    with torch.no_grad():
        for trial in range(32):
            x=torch.randn(4,8,8)
            writes=torch.rand(4,8)>.3
            allowed=torch.rand(4,8)>.2
            a,s=m(x,write_allowed=writes,slot_allowed=allowed)
            b,bs=infer_native(m,x,write_allowed=writes,slot_allowed=allowed,lib=lib)
            err_output=max(err_output,float((a-b).abs().max()))
            err_state=max(err_state,float((s.hv-bs.hv).abs().max()),float((s.hk-bs.hk).abs().max()))
    return {'seed':seed,'episodes':32,'max_output_abs':round(err_output,10),
            'max_state_abs':round(err_state,10),'fla_triton_executed':False,
            'r11_ssc_authority_integrated':False}
