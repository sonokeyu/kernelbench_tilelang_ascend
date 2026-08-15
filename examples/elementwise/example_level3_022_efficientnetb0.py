"""TileLang L3 #22 EfficientNetB0: stem + MBConv x2 + head + GAP + fc."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,relu6_2d,dwcv,ewadd,gap2d,ln
S=torch.npu.synchronize;EPS=1e-5;TK={'TH':4,'TW':4};P1=lambda t:F.pad(t,(1,1,1,1))

def mbconv(x,p,o):
    """pw expand+bn+relu6 -> dw3x3+bn+relu6 -> pw project+bn -> +residual."""
    BS=x.shape[0];C=x.shape[1]
    ew,eb,m1,v1,g1,t1,dw,dwb,m2,v2,g2,t2,pw,pb,m3,v3,g3,t3=p[o:o+18]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS);i3=1.0/torch.sqrt(v3+EPS)
    E=ew.shape[0]
    h=cv(BS,C,E,x.shape[2],x.shape[3],1,1,**TK)(x,ew,eb);S()
    h=bn2d(BS,E,h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,t1);S()
    h=relu6_2d(BS,E,h.shape[2],h.shape[3],**TK)(h);S()
    h2=P1(h);h2=dwcv(BS,E,h2.shape[2],h2.shape[3],3,1,**TK)(h2,dw,dwb);S()
    h2=bn2d(BS,E,h2.shape[2],h2.shape[3],EPS,**TK)(h2,m2,i2,g2,t2);S()
    h2=relu6_2d(BS,E,h2.shape[2],h2.shape[3],**TK)(h2);S()
    h2=cv(BS,E,pw.shape[0],h2.shape[2],h2.shape[3],1,1,**TK)(h2,pw,pb);S()
    h2=bn2d(BS,pw.shape[0],h2.shape[2],h2.shape[3],EPS,**TK)(h2,m3,i3,g3,t3);S()
    if pw.shape[0]==C:
        h2=ewadd(BS,pw.shape[0],h2.shape[2],h2.shape[3],**TK)(h2,x);S()
    return h2

def run(x,p):
    BS=x.shape[0]
    cw,cb,m0,v0,g0,t0=p[0:6];i0=1.0/torch.sqrt(v0+EPS)
    h=P1(x);h=cv(BS,x.shape[1],cw.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,cw,cb);S()
    h=bn2d(BS,cw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m0,i0,g0,t0);S()
    h=relu2d(BS,cw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=mbconv(h,p,6);S()
    h=mbconv(h,p,24);S()
    hw,hb,mh,vh,gh,th=p[42:48];ih=1.0/torch.sqrt(vh+EPS)
    h=cv(BS,h.shape[1],hw.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,hw,hb);S()
    h=bn2d(BS,hw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,mh,ih,gh,th);S()
    h=relu2d(BS,hw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    g=gap2d(BS,h.shape[1],h.shape[2],h.shape[3])(h);S()
    return ln(BS,h.shape[1],p[48].shape[0])(g,p[48],p[49])

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,C,E,HC,NC=2,3,16,8,16,12,4
    x=torch.randn(BS,IC,HW,HW).npu()
    def bnp(c):return [torch.randn(c).npu(),torch.randn(c).abs().npu()+0.1,torch.randn(c).npu(),torch.randn(c).npu()]
    def mbp(C,E):
        return ([torch.randn(E,C,1,1).npu(),torch.randn(E).npu()]+bnp(E)
                +[torch.randn(E,1,3,3).npu(),torch.randn(E).npu()]+bnp(E)
                +[torch.randn(C,E,1,1).npu(),torch.randn(C).npu()]+bnp(C))
    P=[torch.randn(C,IC,3,3).npu(),torch.randn(C).npu()]+bnp(C)+mbp(C,E)+mbp(C,E)
    P+=[torch.randn(HC,C,1,1).npu(),torch.randn(HC).npu()]+bnp(HC)
    P+=[torch.randn(NC,HC).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def bnr(r,o):return F.batch_norm(r,cp[o],cp[o+1],cp[o+2],cp[o+3],eps=e)
    def mbr(r,o):
        s=r;E_=cp[o].shape[0]
        h=F.conv2d(r,cp[o],cp[o+1]);h=F.relu6(bnr(h,o+2))
        h=F.conv2d(F.pad(h,(1,1,1,1)),cp[o+6],bias=cp[o+7],groups=E_);h=F.relu6(bnr(h,o+8))
        h=F.conv2d(h,cp[o+12],cp[o+13]);h=bnr(h,o+14)
        return h+s if h.shape[1]==s.shape[1] else h
    r=F.conv2d(F.pad(xc,(1,1,1,1)),cp[0],cp[1]);r=F.relu(bnr(r,2))
    r=mbr(r,6);r=mbr(r,24)
    r=F.conv2d(r,cp[42],cp[43]);r=F.relu(bnr(r,44))
    r=r.mean(dim=[2,3]);r=F.linear(r,cp[48],cp[49])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_022_efficientnetb0 passed")
