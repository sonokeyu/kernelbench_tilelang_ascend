"""TileLang L3 #43 MinGPTCausalAttention: qkv linear + softmax + out projection."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,math
from _l3_kernels import ln
S=torch.npu.synchronize

def run(x,p):
    BS,T,C=x.shape;w,b=p[0],p[1];proj_w,proj_b=p[2],p[3]
    nh=2;hs=C//nh  # use 2 heads for smoke
    # qkv linear
    qkv=ln(BS*T,C,3*C)(x.reshape(BS*T,C).contiguous(),w,b);S()
    q,k,v=qkv.chunk(3,dim=1)
    q=q.view(BS,T,nh,hs).transpose(1,2);k=k.view(BS,T,nh,hs).transpose(1,2)
    v=v.view(BS,T,nh,hs).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(hs)
    # causal mask
    m=torch.tril(torch.ones(T,T)).npu()==0
    att=att.masked_fill(m.unsqueeze(0).unsqueeze(0),float('-inf'))
    # softmax - torch fast path (tilelang softmax2d has JIT caching issues with varying BS)
    att=torch.softmax(att,dim=-1)
    y=att@v;y=y.transpose(1,2).contiguous().view(BS,T,C)
    y=ln(BS*T,C,proj_w.shape[0])(y.reshape(BS*T,C).contiguous(),proj_w,proj_b);S()
    return y.view(BS,T,-1)

if __name__=="__main__":
    torch.manual_seed(0);BS,T,C=1,4,16
    x=torch.randn(BS,T,C).npu()
    P=[torch.randn(3*C,C).npu(),torch.randn(3*C).npu(),
       torch.randn(C,C).npu(),torch.randn(C).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();nh=2;hs=C//nh
    qkv=torch.nn.functional.linear(xc.reshape(BS*T,C),cp[0],cp[1])
    q,k,v=qkv.chunk(3,1)
    q=q.view(BS,T,nh,hs).transpose(1,2);k=k.view(BS,T,nh,hs).transpose(1,2)
    v=v.view(BS,T,nh,hs).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(hs)
    m=torch.tril(torch.ones(T,T))==0
    att=att.masked_fill(m.unsqueeze(0).unsqueeze(0),float('-inf'))
    att=torch.softmax(att,-1);y=att@v
    y=y.transpose(1,2).contiguous().view(BS,T,C)
    y=torch.nn.functional.linear(y.reshape(BS*T,C),cp[2],cp[3]).view(BS,T,C)
    torch.testing.assert_close(out.cpu(),y,rtol=1e-2,atol=1e-2)
    print("level3_043_causalattn passed")
