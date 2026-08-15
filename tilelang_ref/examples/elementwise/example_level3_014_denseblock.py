"""TileLang L3 #14 DenseBlock: BN+ReLU+conv1x1 + BN+ReLU+conv3x3 + cat, single layer smoke."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import bn2d,relu2d,cv,cat2d
S=torch.npu.synchronize;TK={'TH':8,'TW':8};EPS=1e-5;P1=lambda x:F.pad(x,(1,1,1,1))

def run(x,p):
    BS=x.shape[0];c1w,c1b=p[0],p[1];m1,v1,g1,b1=p[2:6];c2w,c2b=p[6:8];m2,v2,g2,b2=p[8:12]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=bn2d(BS,x.shape[1],x.shape[2],x.shape[3],EPS,**TK)(x,m1,i1,g1,b1);S()
    h=relu2d(BS,x.shape[1],h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,x.shape[1],c1w.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,c1w,c1b);S()
    h2=bn2d(BS,c1w.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,b2);S()
    h2=relu2d(BS,c1w.shape[0],h2.shape[2],h2.shape[3],**TK)(h2);S()
    h2=P1(h2);h2=cv(BS,c1w.shape[0],c2w.shape[0],h2.shape[2],h2.shape[3],3,1,**TK)(h2,c2w,c2b);S()
    return cat2d(BS,x.shape[1],c2w.shape[0],h.shape[2],h.shape[3],**TK)(x,h2);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,C,GR,HW=2,8,4,16
    x=torch.randn(BS,C,HW,HW).npu()
    P=[torch.randn(GR,C,1,1).npu(),torch.randn(GR).npu(),
       torch.randn(C).npu(),torch.randn(C).abs().npu()+0.1,torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(GR,GR,3,3).npu(),torch.randn(GR).npu(),
       torch.randn(GR).npu(),torch.randn(GR).abs().npu()+0.1,torch.randn(GR).npu(),torch.randn(GR).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=F.batch_norm(xc,cp[2],cp[3],cp[4],cp[5],eps=e);r=F.relu(r)
    r=F.conv2d(r,cp[0],cp[1]);r2=F.batch_norm(r,cp[8],cp[9],cp[10],cp[11],eps=e);r2=F.relu(r2)
    r2=F.conv2d(F.pad(r2,(1,1,1,1)),cp[6],cp[7]);r=torch.cat([xc,r2],1)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_014_denseblock passed")
