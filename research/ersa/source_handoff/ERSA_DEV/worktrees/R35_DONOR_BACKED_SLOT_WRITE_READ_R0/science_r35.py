"""R35 tiny two-token write/read probe; not a language-model benchmark."""
import torch
from r35.slot_write import ERSASlotWriteRead


def one_seed(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=ERSASlotWriteRead(d_model=8,num_slots=8,topk=2,max_steps=4)
    train_v=torch.randn(16,8)*.6
    hold_v=torch.randn(16,8)*.6
    def samples(v):
        return torch.stack((v,torch.zeros_like(v)),dim=1)
    allowed=torch.tensor([True,False]).expand(16,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.015)
    with torch.no_grad():
        start_y,_=model(samples(train_v),write_allowed=allowed)
        initial=float((start_y[:,1]-train_v).square().mean())
    for _ in range(90):
        optimizer.zero_grad(set_to_none=True)
        y,_=model(samples(train_v),write_allowed=allowed)
        loss=(y[:,1]-train_v).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y,_=model(samples(train_v),write_allowed=allowed)
        end=float((y[:,1]-train_v).square().mean())
        yy,_=model(samples(hold_v),write_allowed=allowed)
        heldout=float((yy[:,1]-hold_v).square().mean())
        no,_=model(samples(hold_v),write_allowed=torch.zeros_like(allowed))
        no_write=float((no[:,1]-hold_v).square().mean())
    return dict(seed=seed,train_first=round(initial,8),train_last=round(end,8),
                heldout=round(heldout,8),no_write=round(no_write,8),steps=90,
                checkpoint_writes=0,model_scope='TWO_TOKEN_WRITE_READ_PROBE')
