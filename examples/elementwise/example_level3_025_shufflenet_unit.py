"""TileLang L3 #25 ShuffleNetUnit: gconv1x1+bn+relu + shuffle + dw3x3+bn."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import gcv1x1,bn2d,relu2d,dwcv
S=torch.npu.synchronize;EPS=1e-5

def run(x,p):
    BS,C,H,W=x.shape[0],x.shape[1],x.shape[2],x.shape[3];G=2;OC=C*2
    g1w,g1b,m1,v1,gp,bp,dw,dwb,m2,v2,g2p,b2p=p
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=gcv1x1(BS,C,OC,G,H,W)(x,g1w,g1b);S()
    h=bn2d(BS,OC,H,W,EPS)(h,m1,i1,gp,bp);S()
    h=relu2d(BS,OC,H,W)(h);S()
    h=h.view(BS,G,-1,H,W).transpose(1,2).contiguous().view(BS,OC,H,W)
    h2=torch.nn.functional.pad(h,(1,1,1,1))
    h2=dwcv(BS,OC,h2.shape[2],h2.shape[3],3,1)(h2,dw,dwb);S()
    h2=bn2d(BS,OC,h2.shape[2],h2.shape[3],EPS)(h2,m2,i2,g2p,b2p);S()
    return h2

if __name__=="__main__":
    torch.manual_seed(0);BS,C,G,HW=2,16,2,16;OC=C*2
    x=torch.randn(BS,C,HW,HW).npu()
    P=[torch.randn(OC,C//G,1,1).npu(),torch.randn(OC).npu(),
       torch.randn(OC).npu(),torch.randn(OC).abs().npu()+0.1,torch.randn(OC).npu(),torch.randn(OC).npu(),
       torch.randn(OC,1,3,3).npu(),torch.randn(OC).npu(),
       torch.randn(OC).npu(),torch.randn(OC).abs().npu()+0.1,torch.randn(OC).npu(),torch.randn(OC).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=torch.nn.functional.conv2d(xc,cp[0],cp[1],groups=G)
    r=torch.nn.functional.batch_norm(r,cp[2],cp[3],cp[4],cp[5],eps=e);r=torch.nn.functional.relu(r)
    r=r.view(BS,G,-1,HW,HW).transpose(1,2).contiguous().view(BS,OC,HW,HW)
    r=torch.nn.functional.conv2d(torch.nn.functional.pad(r,(1,1,1,1)),cp[6],bias=cp[7],groups=OC)
    r=torch.nn.functional.batch_norm(r,cp[8],cp[9],cp[10],cp[11],eps=e)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_025_shufflenet_unit passed")
