"""TileLang L3 #10 ResNet101: conv1+bn+relu+pool + Bottleneck x2 + GAP + fc."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,pool,ewadd,gap2d,ln
S=torch.npu.synchronize;EPS=1e-5;TK={'TH':4,'TW':4};P1=lambda t:F.pad(t,(1,1,1,1))

def bottleneck(x,p,o):
    """1x1+bn+relu -> 3x3+bn+relu -> 1x1+bn -> +residual -> relu. IC==OC."""
    BS=x.shape[0];C=x.shape[1]
    c1,b1,m1,v1,g1,t1,c2,b2,m2,v2,g2,t2,c3,b3,m3,v3,g3,t3=p[o:o+18]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS);i3=1.0/torch.sqrt(v3+EPS)
    h=cv(BS,C,c1.shape[0],x.shape[2],x.shape[3],1,1,**TK)(x,c1,b1);S()
    h=bn2d(BS,c1.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,t1);S()
    h=relu2d(BS,c1.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=P1(h);h=cv(BS,c1.shape[0],c2.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,c2,b2);S()
    h=bn2d(BS,c2.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,t2);S()
    h=relu2d(BS,c2.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,c2.shape[0],c3.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,c3,b3);S()
    h=bn2d(BS,c3.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m3,i3,g3,t3);S()
    h=ewadd(BS,c3.shape[0],h.shape[2],h.shape[3],**TK)(h,x);S()
    return relu2d(BS,c3.shape[0],h.shape[2],h.shape[3],**TK)(h)

def run(x,p):
    BS=x.shape[0]
    cw,cb,m0,v0,g0,t0=p[0:6];i0=1.0/torch.sqrt(v0+EPS)
    h=P1(x);h=cv(BS,x.shape[1],cw.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,cw,cb);S()
    h=bn2d(BS,cw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m0,i0,g0,t0);S()
    h=relu2d(BS,cw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=pool(BS,cw.shape[0],h.shape[2],h.shape[3],2,2,**TK)(h);S()
    h=bottleneck(h,p,6);S()
    h=bottleneck(h,p,24);S()
    g=gap2d(BS,h.shape[1],h.shape[2],h.shape[3])(h);S()
    return ln(BS,h.shape[1],p[42].shape[0])(g,p[42],p[43])

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,C,NC=2,3,16,8,4
    x=torch.randn(BS,IC,HW,HW).npu()
    def bnp(c):return [torch.randn(c).npu(),torch.randn(c).abs().npu()+0.1,torch.randn(c).npu(),torch.randn(c).npu()]
    def bp(C):
        r=[torch.randn(C,C,1,1).npu(),torch.randn(C).npu()]+bnp(C)
        r+=[torch.randn(C,C,3,3).npu(),torch.randn(C).npu()]+bnp(C)
        r+=[torch.randn(C,C,1,1).npu(),torch.randn(C).npu()]+bnp(C)
        return r
    P=[torch.randn(C,IC,3,3).npu(),torch.randn(C).npu()]+bnp(C)+bp(C)+bp(C)
    P+=[torch.randn(NC,C).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def bnr(r,o):return F.batch_norm(r,cp[o],cp[o+1],cp[o+2],cp[o+3],eps=e)
    def bref(r,o):
        s=r
        r=F.conv2d(r,cp[o],cp[o+1]);r=bnr(r,o+2);r=F.relu(r)
        r=F.conv2d(F.pad(r,(1,1,1,1)),cp[o+6],cp[o+7]);r=bnr(r,o+8);r=F.relu(r)
        r=F.conv2d(r,cp[o+12],cp[o+13]);r=bnr(r,o+14)
        return F.relu(r+s)
    r=F.conv2d(F.pad(xc,(1,1,1,1)),cp[0],cp[1]);r=bnr(r,2);r=F.relu(r)
    r=F.max_pool2d(r,2,2)
    r=bref(r,6);r=bref(r,24)
    r=r.mean(dim=[2,3]);r=F.linear(r,cp[42],cp[43])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_010_resnet101 passed")
