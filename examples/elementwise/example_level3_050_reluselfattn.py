"""TileLang L3 #50 ReLUSelfAttention. Linear qkv + relu attn (no softmax)."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,math
from _l3_kernels import ln,relu2d
S=torch.npu.synchronize

def run(x,p):
    w,b=p;BS,T,C=x.shape;HD=C
    qkv=ln(BS*T,C,3*C)(x.reshape(BS*T,C).contiguous(),w,b);S()
    qkv=qkv.view(BS,T,3,C);q,k,v=qkv[:,:,0,:],qkv[:,:,1,:],qkv[:,:,2,:]
    # Multi-head: (B,nh,T,hs), using 2 heads
    nh=2;hs=C//nh
    q=q.view(BS,T,nh,hs).transpose(1,2);k=k.view(BS,T,nh,hs).transpose(1,2)
    v=v.view(BS,T,nh,hs).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(hs)
    # relu via 4D reshape
    AT=att.reshape(-1).view(BS*nh,1,T,T).contiguous()
    AT=relu2d(BS*nh,1,T,T)(AT);S()
    att=AT.view(BS,nh,T,T)
    y=att@v;y=y.transpose(1,2).contiguous().view(BS,T,C)
    return y

if __name__=="__main__":
    torch.manual_seed(0);BS,T,C=1,4,16
    x=torch.randn(BS,T,C).npu()
    P=[torch.randn(3*C,C).npu(),torch.randn(3*C).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    qkv=torch.nn.functional.linear(xc,cp[0],cp[1]).view(BS,T,3,C)
    q,k,v=qkv[:,:,0,:],qkv[:,:,1,:],qkv[:,:,2,:];nh=2;hs=C//nh
    q=q.view(BS,T,nh,hs).transpose(1,2);k=k.view(BS,T,nh,hs).transpose(1,2)
    v=v.view(BS,T,nh,hs).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(hs)
    att=torch.relu(att)
    y=att@v;y=y.transpose(1,2).contiguous().view(BS,T,C)
    torch.testing.assert_close(out.cpu(),y,rtol=5e-2,atol=5e-2)
    print("level3_050_reluselfattn passed")
