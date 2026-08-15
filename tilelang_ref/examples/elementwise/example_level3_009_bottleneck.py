"""TileLang L3 #9 ResNet Bottleneck. 1x1+bn+relu → 3x3+bn+relu → 1x1+bn → residual_add+relu."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,ewadd
S=torch.npu.synchronize;TK={'TH':8,'TW':8};EPS=1e-5

def run(x,p):
    BS=x.shape[0]
    c1w,c1b,c2w,c2b,c3w,c3b=p[:6]
    m1,v1,g1,b1,m2,v2,g2,b2,m3,v3,g3,b3=p[6:18]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS);i3=1.0/torch.sqrt(v3+EPS)
    IC,OC=x.shape[1],c1w.shape[0];EXP=c3w.shape[0]
    h=cv(BS,IC,OC,x.shape[2],x.shape[3],1,1,**TK)(x,c1w,c1b);S()
    h=bn2d(BS,OC,h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,b1);S()
    h=relu2d(BS,OC,h.shape[2],h.shape[3],**TK)(h);S()
    h2=F.pad(h,(1,1,1,1))
    h2=cv(BS,OC,OC,h2.shape[2],h2.shape[3],3,1,**TK)(h2,c2w,c2b);S()
    h2=bn2d(BS,OC,h2.shape[2],h2.shape[3],EPS,**TK)(h2,m2,i2,g2,b2);S()
    h2=relu2d(BS,OC,h2.shape[2],h2.shape[3],**TK)(h2);S()
    h3=cv(BS,OC,EXP,h2.shape[2],h2.shape[3],1,1,**TK)(h2,c3w,c3b);S()
    h3=bn2d(BS,EXP,h3.shape[2],h3.shape[3],EPS,**TK)(h3,m3,i3,g3,b3);S()
    # residual: if channels differ, we'd need 1x1 conv projection (not in this smoke test)
    h=ewadd(BS,EXP,h3.shape[2],h3.shape[3],**TK)(h3,x);S()
    return relu2d(BS,EXP,h.shape[2],h.shape[3],**TK)(h);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,C,HW=2,64,32  # IC=EXP=C for clean residual
    x=torch.randn(BS,C,HW,HW).npu()
    P=[torch.randn(C,C,1,1).npu(),torch.randn(C).npu(),
       torch.randn(C,C,3,3).npu(),torch.randn(C).npu(),
       torch.randn(C,C,1,1).npu(),torch.randn(C).npu()]
    for _ in range(3):
        P.append(torch.randn(C).npu());P.append(torch.randn(C).abs().npu()+0.1)
        P.append(torch.randn(C).npu());P.append(torch.randn(C).npu())
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=F.conv2d(xc,cp[0],cp[1]);r=F.batch_norm(r,cp[6],cp[7],cp[8],cp[9],eps=e);r=F.relu(r)
    r=F.conv2d(F.pad(r,(1,1,1,1)),cp[2],cp[3])
    r=F.batch_norm(r,cp[10],cp[11],cp[12],cp[13],eps=e);r=F.relu(r)
    r=F.conv2d(r,cp[4],cp[5]);r=F.batch_norm(r,cp[14],cp[15],cp[16],cp[17],eps=e)
    r=r+xc;r=F.relu(r)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_009_bottleneck passed")
