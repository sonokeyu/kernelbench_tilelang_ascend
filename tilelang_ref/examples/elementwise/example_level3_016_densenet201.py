"""TileLang L3 #16 DenseNet201: stem + DenseBlock + Transition + GAP + fc."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,pool,cat2d,gap2d,ln
S=torch.npu.synchronize;EPS=1e-5;TK={'TH':4,'TW':4};P1=lambda t:F.pad(t,(1,1,1,1))

def dlayer(x,p,o):
    """bn+relu+conv1x1 -> bn+relu+conv3x3 -> cat."""
    BS=x.shape[0];C=x.shape[1]
    m1,v1,g1,t1,c1,b1,m2,v2,g2,t2,c2,b2=p[o:o+12]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=bn2d(BS,C,x.shape[2],x.shape[3],EPS,**TK)(x,m1,i1,g1,t1);S()
    h=relu2d(BS,C,h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,C,c1.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,c1,b1);S()
    h2=bn2d(BS,c1.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,t2);S()
    h2=relu2d(BS,c1.shape[0],h2.shape[2],h2.shape[3],**TK)(h2);S()
    h2=P1(h2);h2=cv(BS,c1.shape[0],c2.shape[0],h2.shape[2],h2.shape[3],3,1,**TK)(h2,c2,b2);S()
    return cat2d(BS,C,c2.shape[0],x.shape[2],x.shape[3],**TK)(x,h2)

def run(x,p):
    BS=x.shape[0]
    cw,cb,m0,v0,g0,t0=p[0:6];i0=1.0/torch.sqrt(v0+EPS)
    h=P1(x);h=cv(BS,x.shape[1],cw.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,cw,cb);S()
    h=bn2d(BS,cw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m0,i0,g0,t0);S()
    h=relu2d(BS,cw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=dlayer(h,p,6);S()
    h=dlayer(h,p,18);S()
    # transition: bn+relu+conv1x1+pool
    mt,vt,gt,tt,ctw,ctb=p[30:36];it=1.0/torch.sqrt(vt+EPS)
    h=bn2d(BS,h.shape[1],h.shape[2],h.shape[3],EPS,**TK)(h,mt,it,gt,tt);S()
    h=relu2d(BS,h.shape[1],h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,h.shape[1],ctw.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,ctw,ctb);S()
    h=pool(BS,ctw.shape[0],h.shape[2],h.shape[3],2,2,**TK)(h);S()
    g=gap2d(BS,h.shape[1],h.shape[2],h.shape[3])(h);S()
    return ln(BS,h.shape[1],p[36].shape[0])(g,p[36],p[37])

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,C0,GR,NC=2,3,16,8,4,4
    x=torch.randn(BS,IC,HW,HW).npu()
    def bnp(c):return [torch.randn(c).npu(),torch.randn(c).abs().npu()+0.1,torch.randn(c).npu(),torch.randn(c).npu()]
    P=[torch.randn(C0,IC,3,3).npu(),torch.randn(C0).npu()]+bnp(C0)
    P+=bnp(C0)+[torch.randn(GR,C0,1,1).npu(),torch.randn(GR).npu()]+bnp(GR)+[torch.randn(GR,GR,3,3).npu(),torch.randn(GR).npu()]
    C1=C0+GR
    P+=bnp(C1)+[torch.randn(GR,C1,1,1).npu(),torch.randn(GR).npu()]+bnp(GR)+[torch.randn(GR,GR,3,3).npu(),torch.randn(GR).npu()]
    C2=C1+GR
    CT=8
    P+=bnp(C2)+[torch.randn(CT,C2,1,1).npu(),torch.randn(CT).npu()]
    P+=[torch.randn(NC,CT).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def bnr(r,o):return F.batch_norm(r,cp[o],cp[o+1],cp[o+2],cp[o+3],eps=e)
    def dl(r,o):
        h=F.relu(bnr(r,o));h=F.conv2d(h,cp[o+4],cp[o+5])
        h2=F.relu(bnr(h,o+6));h2=F.conv2d(F.pad(h2,(1,1,1,1)),cp[o+10],cp[o+11])
        return torch.cat([r,h2],1)
    r=F.conv2d(F.pad(xc,(1,1,1,1)),cp[0],cp[1]);r=F.relu(bnr(r,2))
    r=dl(r,6);r=dl(r,18)
    r=F.relu(bnr(r,30));r=F.conv2d(r,cp[34],cp[35]);r=F.max_pool2d(r,2,2)
    r=r.mean(dim=[2,3]);r=F.linear(r,cp[36],cp[37])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_015_densenet201 passed")
