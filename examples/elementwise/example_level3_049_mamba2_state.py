"""TileLang L3 #49 Mamba2ReturnFinalState: SSD op returning final state."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import exp2d
S=torch.npu.synchronize

def tl_exp(t):
    sh=t.shape;n=t.numel()
    r=exp2d(n,1,1,1)(t.reshape(n,1,1,1).contiguous());S()
    return r.reshape(sh)

def segsum(x):
    T=x.size(-1)
    xc=torch.cumsum(x,dim=-1)
    xs=xc[...,:,None]-xc[...,None,:]
    m=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=0)
    return xs.masked_fill(~m,-torch.inf)

def run(X,A,B,C,block_len):
    from einops import rearrange
    Xb,Ab,Bb,Cb=[rearrange(t,"b (c l) ... -> b c l ...",l=block_len) for t in (X,A,B,C)]
    Ab=rearrange(Ab,"b c l h -> b h c l")
    Ac=torch.cumsum(Ab,dim=-1)
    ds=tl_exp(Ac[:,:,:,-1:]-Ac)
    st=torch.einsum("bclhn,bhcl,bclhp->bchpn",Bb,ds,Xb)
    st=torch.cat([torch.zeros_like(st[:,:1]),st],dim=1)
    dc=tl_exp(segsum(F.pad(Ac[:,:,:,-1],(1,0))))
    return torch.einsum("bhzc,bchpn->bzhpn",dc,st)[:,-1]

if __name__=="__main__":
    torch.manual_seed(0);BS,SL,NH,DH,DS,BL=2,8,2,4,4,4
    X=torch.randn(BS,SL,NH,DH).npu()
    A=torch.randn(BS,SL,NH).npu()*0.1
    B=torch.randn(BS,SL,NH,DS).npu()
    C=torch.randn(BS,SL,NH,DS).npu()
    out=run(X,A,B,C,BL)
    Xc,Acp,Bc,Cc=X.cpu(),A.cpu(),B.cpu(),C.cpu()
    from einops import rearrange
    Xb,Ab,Bb,Cb=[rearrange(t,"b (c l) ... -> b c l ...",l=BL) for t in (Xc,Acp,Bc,Cc)]
    Ab=rearrange(Ab,"b c l h -> b h c l")
    Acum=torch.cumsum(Ab,dim=-1)
    ds=torch.exp(Acum[:,:,:,-1:]-Acum)
    st=torch.einsum("bclhn,bhcl,bclhp->bchpn",Bb,ds,Xb)
    st=torch.cat([torch.zeros_like(st[:,:1]),st],dim=1)
    dc=torch.exp(segsum(F.pad(Acum[:,:,:,-1],(1,0))))
    ref=torch.einsum("bhzc,bchpn->bzhpn",dc,st)[:,-1]
    torch.testing.assert_close(out.cpu(),ref,rtol=5e-2,atol=5e-2)
    print("level3_049_mamba2_state passed")
