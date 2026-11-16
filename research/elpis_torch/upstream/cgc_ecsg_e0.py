"""ECS_G E0 microscopic cubic cell — float64 differentiable research reference.

Derived from pinned R97M C and NumPy mathematics. Episode-local differentiable
W only: no K1, FMS, persistence, transactions, or trained neural expert claim.
"""
import torch


def require_float64(w, x=None):
    if not isinstance(w, torch.Tensor) or w.ndim != 2 or w.dtype != torch.float64:
        raise ValueError('W_REQUIRES_FLOAT64_DxN')
    if x is not None and (not isinstance(x, torch.Tensor) or x.ndim != 2
                          or x.dtype != torch.float64 or x.shape[1] != w.shape[0]
                          or x.device != w.device):
        raise ValueError('X_REQUIRES_FLOAT64_BxD_SAME_DEVICE')
    if w.shape[0] < 1 or w.shape[1] < 1:
        raise ValueError('POSITIVE_DIM_WIDTH_REQUIRED')
    if not torch.isfinite(w).all() or (x is not None and not torch.isfinite(x).all()):
        raise ValueError('NONFINITE_INPUT')


def phi(z):
    return 0.5 * z + 0.5 * z*z + 0.5 * z*z*z


def phip(z):
    return 0.5 + z + 1.5*z*z


def forward(w, x):
    require_float64(w, x)
    return phi(x @ w).sum(dim=1)


def project_s3(w):
    require_float64(w)
    d = w.shape[0]
    mu = w.sum(dim=1)
    M = w @ w.T
    T3 = torch.einsum('ai,bi,ci->abc', w, w, w)
    ii, jj = torch.triu_indices(d, d, device=w.device)
    tr = torch.tensor([(a,b,c) for a in range(d) for b in range(a,d) for c in range(b,d)],dtype=torch.long,device=w.device)
    return mu, M[ii,jj], T3[tr[:,0],tr[:,1],tr[:,2]]


def forward_s3(mu, m_packed, t_packed, x):
    if (mu.ndim != 1 or x.ndim != 2 or x.shape[1] != mu.shape[0]
        or any(z.dtype != torch.float64 or z.device != mu.device for z in (mu,m_packed,t_packed,x))):
        raise ValueError('S3_BAD_SHAPES_OR_TYPES')
    d = mu.shape[0]
    if m_packed.numel() != d*(d+1)//2 or t_packed.numel() != d*(d+1)*(d+2)//6:
        raise ValueError('S3_PACKED_LENGTH')
    if not all(torch.isfinite(z).all() for z in (mu,m_packed,t_packed,x)):
        raise ValueError('NONFINITE_INPUT')
    linear = x @ mu
    quadratic = torch.zeros_like(linear)
    cubic = torch.zeros_like(linear)
    p = 0
    for a in range(d):
        for b in range(a,d):
            quadratic = quadratic + (1.0 if a==b else 2.0)*m_packed[p]*x[:,a]*x[:,b]
            p += 1
    p = 0
    for a in range(d):
        for b in range(a,d):
            for c in range(b,d):
                k=1.0 if a==b==c else (3.0 if a==b or b==c or a==c else 6.0)
                cubic = cubic+k*t_packed[p]*x[:,a]*x[:,b]*x[:,c]
                p+=1
    return 0.5*(linear+quadratic+cubic)


def mse_loss(w,x,y):
    p=forward(w,x)
    if y.ndim != 1 or y.shape != p.shape or y.dtype != torch.float64:
        raise ValueError('Y_REQUIRES_FLOAT64_B')
    return ((p-y)**2).mean()


def analytic_grad(w,x,y):
    e=forward(w,x)-y
    z=x@w
    return (2.0/x.shape[0]) * x.T @ (e[:,None]*phip(z))


def gd_step(w,x,y,learning_rate):
    return w-learning_rate*analytic_grad(w,x,y)


class ECSGE0Cell(torch.nn.Module):
    def __init__(self, initial_w):
        super().__init__()
        require_float64(initial_w)
        self.w=torch.nn.Parameter(initial_w.detach().clone())
    def forward(self,x):
        return forward(self.w,x)
