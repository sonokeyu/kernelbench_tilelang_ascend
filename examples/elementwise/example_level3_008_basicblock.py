"""TileLang L3 #8 BasicBlock. conv+bn+relu x2 + residual add + relu."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,ewadd
S=torch.npu.synchronize;P1=lambda x:F.pad(x,(1,1,1,1))
TK={'TH':8,'TW':8};EPS=1e-5

def run(x,p):
    BS=x.shape[0];eps=1e-5
    c1w,c1b,c2w,c2b=p[:4]
    bn1m,bn1v,bn1g,bn1b,bn2m,bn2v,bn2g,bn2b=p[4:12]
    i1=1.0/torch.sqrt(bn1v+eps);i2=1.0/torch.sqrt(bn2v+eps)
    h=P1(x);h=cv(BS,x.shape[1],c1w.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,c1w,c1b);S()
    h=bn2d(BS,c1w.shape[0],h.shape[2],h.shape[3],eps,**TK)(h,bn1m,i1,bn1g,bn1b);S()
    h=relu2d(BS,c1w.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h2=P1(h);h2=cv(BS,c1w.shape[0],c2w.shape[0],h2.shape[2],h2.shape[3],3,1,**TK)(h2,c2w,c2b);S()
    h2=bn2d(BS,c2w.shape[0],h2.shape[2],h2.shape[3],eps,**TK)(h2,bn2m,i2,bn2g,bn2b);S()
    h=ewadd(BS,c2w.shape[0],h2.shape[2],h2.shape[3],**TK)(h2,x);S()
    return relu2d(BS,c2w.shape[0],h.shape[2],h.shape[3],**TK)(h);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,OC,HW=2,64,64,32  # IC==OC for matching residual
    x=torch.randn(BS,IC,HW,HW).npu()
    e=1e-5
    P=[torch.randn(OC,IC,3,3).npu(),torch.randn(OC).npu(),
       torch.randn(OC,OC,3,3).npu(),torch.randn(OC).npu()]
    for _ in range(2):
        P.append(torch.randn(OC).npu());P.append(torch.randn(OC).abs().npu()+0.1)
        P.append(torch.randn(OC).npu());P.append(torch.randn(OC).npu())
    out=run(x,P)
    cp=[p.cpu() for p in P]
    r=F.conv2d(F.pad(x.cpu(),(1,1,1,1)),cp[0],cp[1])
    r=F.batch_norm(r,cp[4],cp[5],cp[6],cp[7],eps=e);r=F.relu(r)
    r=F.conv2d(F.pad(r,(1,1,1,1)),cp[2],cp[3])
    r=F.batch_norm(r,cp[8],cp[9],cp[10],cp[11],eps=e)
    r=r+x.cpu();r=F.relu(r)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_008_basicblock passed")
