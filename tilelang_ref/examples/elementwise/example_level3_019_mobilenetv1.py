"""TileLang L3 #19 MobileNetV1 block: dw3x3+bn+relu + pw1x1+bn+relu."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import dwcv,bn2d,relu2d,cv
S=torch.npu.synchronize;TK={'TH':8,'TW':8};EPS=1e-5;P1=lambda x:F.pad(x,(1,1,1,1))

def run(x,p):
    BS=x.shape[0];dw,dwb,m1,v1,g1,b1,pw,pwb,m2,v2,g2,b2=p
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=P1(x);h=dwcv(BS,x.shape[1],h.shape[2],h.shape[3],3,1,**TK)(h,dw,dwb);S()
    h=bn2d(BS,x.shape[1],h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,b1);S()
    h=relu2d(BS,x.shape[1],h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,x.shape[1],pw.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,pw,pwb);S()
    h=bn2d(BS,pw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,b2);S()
    return relu2d(BS,pw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,C,HW=2,32,32
    x=torch.randn(BS,C,HW,HW).npu()
    P=[torch.randn(C,1,3,3).npu(),torch.randn(C).npu(),
       torch.randn(C).npu(),torch.randn(C).abs().npu()+0.1,torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(64,C,1,1).npu(),torch.randn(64).npu(),
       torch.randn(64).npu(),torch.randn(64).abs().npu()+0.1,torch.randn(64).npu(),torch.randn(64).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=F.conv2d(F.pad(xc,(1,1,1,1)),cp[0],bias=cp[1],groups=C)
    r=F.batch_norm(r,cp[2],cp[3],cp[4],cp[5],eps=e);r=F.relu(r)
    r=F.conv2d(r,cp[6],bias=cp[7]);r=F.batch_norm(r,cp[8],cp[9],cp[10],cp[11],eps=e);r=F.relu(r)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_019_mobilenetv1 passed")
