"""TileLang L3 #20 MobileNetV2 inverted residual."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu6_2d,dwcv,ewadd
S=torch.npu.synchronize;TK={'TH':8,'TW':8};EPS=1e-5;P1=lambda x:F.pad(x,(1,1,1,1))

def run(x,p):
    BS=x.shape[0]
    pw1,pwb1,m1,v1,g1,b1,dw,dwb,m2,v2,g2,b2,pw2,pwb2,m3,v3,g3,b3=p
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS);i3=1.0/torch.sqrt(v3+EPS)
    EXP=pw1.shape[0]
    h=cv(BS,x.shape[1],EXP,x.shape[2],x.shape[3],1,1,**TK)(x,pw1,pwb1);S()
    h=bn2d(BS,EXP,h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,b1);S()
    h=relu6_2d(BS,EXP,h.shape[2],h.shape[3],**TK)(h);S()
    h2=P1(h);h2=dwcv(BS,EXP,h2.shape[2],h2.shape[3],3,1,**TK)(h2,dw,dwb);S()
    h2=bn2d(BS,EXP,h2.shape[2],h2.shape[3],EPS,**TK)(h2,m2,i2,g2,b2);S()
    h2=relu6_2d(BS,EXP,h2.shape[2],h2.shape[3],**TK)(h2);S()
    h2=cv(BS,EXP,pw2.shape[0],h2.shape[2],h2.shape[3],1,1,**TK)(h2,pw2,pwb2);S()
    h2=bn2d(BS,pw2.shape[0],h2.shape[2],h2.shape[3],EPS,**TK)(h2,m3,i3,g3,b3);S()
    return ewadd(BS,pw2.shape[0],h2.shape[2],h2.shape[3],**TK)(h2,x);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,C,HW,EXP=2,16,32,64;OC=C
    x=torch.randn(BS,C,HW,HW).npu()
    # 18 params: pw, bn1x4, dw, bn2x4, proj, bn3x4
    P=[torch.randn(EXP,C,1,1).npu(),torch.randn(EXP).npu(),
       torch.randn(EXP).npu(),torch.randn(EXP).abs().npu()+0.1,torch.randn(EXP).npu(),torch.randn(EXP).npu(),
       torch.randn(EXP,1,3,3).npu(),torch.randn(EXP).npu(),
       torch.randn(EXP).npu(),torch.randn(EXP).abs().npu()+0.1,torch.randn(EXP).npu(),torch.randn(EXP).npu(),
       torch.randn(OC,EXP,1,1).npu(),torch.randn(OC).npu(),
       torch.randn(OC).npu(),torch.randn(OC).abs().npu()+0.1,torch.randn(OC).npu(),torch.randn(OC).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=F.conv2d(xc,cp[0],cp[1]);r=F.batch_norm(r,cp[2],cp[3],cp[4],cp[5],eps=e);r=F.relu6(r)
    r=F.conv2d(F.pad(r,(1,1,1,1)),cp[6],bias=cp[7],groups=EXP)
    r=F.batch_norm(r,cp[8],cp[9],cp[10],cp[11],eps=e);r=F.relu6(r)
    r=F.conv2d(r,cp[12],bias=cp[13]);r=F.batch_norm(r,cp[14],cp[15],cp[16],cp[17],eps=e)
    r=r+xc
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_020_mobilenetv2 passed")
