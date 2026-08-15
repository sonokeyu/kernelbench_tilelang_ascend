"""TileLang L3 #32 cvt: tilelang linear for attention, torch softmax."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,math
from _l3_kernels import ln
S=torch.npu.synchronize
def run(x,p):
    BS,C,H,W=x.shape;L=H*W;NH=2;HS=C//NH;ipw,ipb,opw,opb=p
    x2d=x.reshape(BS,C,L).transpose(1,2).contiguous().reshape(BS*L,C)
    qkv=ln(BS*L,C,3*C)(x2d,ipw,ipb);S()
    q,k,v=qkv.chunk(3,1)
    q=q.view(BS,L,NH,HS).transpose(1,2);k=k.view(BS,L,NH,HS).transpose(1,2)
    v=v.view(BS,L,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS);att=torch.softmax(att,-1)
    y=att@v;y=y.transpose(1,2).contiguous().view(BS*L,C)
    y=ln(BS*L,C,C)(y,opw,opb);S()
    return y.view(BS,C,H,W)
if __name__=="__main__":
    torch.manual_seed(0);BS,C,H,W=2,8,4,4;NH=2;HS=C//NH
    x=torch.randn(BS,C,H,W).npu()
    P=[torch.randn(3*C,C).npu(),torch.randn(3*C).npu(),torch.randn(C,C).npu(),torch.randn(C).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    x2d=xc.reshape(BS,C,H*W).transpose(1,2).reshape(BS*H*W,C)
    qkv=torch.nn.functional.linear(x2d,cp[0],cp[1]);q,k,v=qkv.chunk(3,1)
    q=q.view(BS,H*W,NH,HS).transpose(1,2);k=k.view(BS,H*W,NH,HS).transpose(1,2)
    v=v.view(BS,H*W,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS);att=torch.softmax(att,-1)
    y=att@v;y=y.transpose(1,2).contiguous().view(BS*H*W,C)
    y=torch.nn.functional.linear(y,cp[2],cp[3]).view(BS,C,H,W)
    torch.testing.assert_close(out.cpu(),y,rtol=5e-2,atol=5e-2)
    print("level3_032 cvt passed")
