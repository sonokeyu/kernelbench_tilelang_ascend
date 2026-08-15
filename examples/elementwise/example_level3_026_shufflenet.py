"""TileLang L3 #26 ShuffleNet: stem + ShuffleNetUnit x2 + conv5 + GAP + fc."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,gcv1x1,bn2d,relu2d,pool,dwcv,gap2d,ln
S=torch.npu.synchronize;EPS=1e-5;TK={'TH':4,'TW':4};P1=lambda t:F.pad(t,(1,1,1,1))

def sunit(x,p,o,G=2):
    """gconv1x1+bn+relu -> shuffle -> dw3x3+bn -> gconv1x1+bn."""
    BS,C,H,W=x.shape
    g1w,g1b,m1,v1,gp1,bp1,dw,dwb,m2,v2,gp2,bp2,g2w,g2b,m3,v3,gp3,bp3=p[o:o+18]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS);i3=1.0/torch.sqrt(v3+EPS)
    MC=g1w.shape[0]
    h=gcv1x1(BS,C,MC,G,H,W,**TK)(x,g1w,g1b);S()
    h=bn2d(BS,MC,H,W,EPS,**TK)(h,m1,i1,gp1,bp1);S()
    h=relu2d(BS,MC,H,W,**TK)(h);S()
    h=h.view(BS,G,-1,H,W).transpose(1,2).contiguous().view(BS,MC,H,W)
    h2=P1(h);h2=dwcv(BS,MC,h2.shape[2],h2.shape[3],3,1,**TK)(h2,dw,dwb);S()
    h2=bn2d(BS,MC,h2.shape[2],h2.shape[3],EPS,**TK)(h2,m2,i2,gp2,bp2);S()
    h2=gcv1x1(BS,MC,g2w.shape[0],G,h2.shape[2],h2.shape[3],**TK)(h2,g2w,g2b);S()
    return bn2d(BS,g2w.shape[0],h2.shape[2],h2.shape[3],EPS,**TK)(h2,m3,i3,gp3,bp3)

def run(x,p):
    BS=x.shape[0]
    cw,cb,m0,v0,g0,t0=p[0:6];i0=1.0/torch.sqrt(v0+EPS)
    h=P1(x);h=cv(BS,x.shape[1],cw.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,cw,cb);S()
    h=bn2d(BS,cw.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m0,i0,g0,t0);S()
    h=relu2d(BS,cw.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=pool(BS,cw.shape[0],h.shape[2],h.shape[3],2,2,**TK)(h);S()
    h=sunit(h,p,6);S()
    h=sunit(h,p,24);S()
    c5w,c5b,m5,v5,g5,t5=p[42:48];i5=1.0/torch.sqrt(v5+EPS)
    h=cv(BS,h.shape[1],c5w.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,c5w,c5b);S()
    h=bn2d(BS,c5w.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m5,i5,g5,t5);S()
    h=relu2d(BS,c5w.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    g=gap2d(BS,h.shape[1],h.shape[2],h.shape[3])(h);S()
    return ln(BS,h.shape[1],p[48].shape[0])(g,p[48],p[49])

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,C,MC,C5,NC,G=2,3,16,8,8,12,4,2
    x=torch.randn(BS,IC,HW,HW).npu()
    def bnp(c):return [torch.randn(c).npu(),torch.randn(c).abs().npu()+0.1,torch.randn(c).npu(),torch.randn(c).npu()]
    def sup(C,MC):
        return ([torch.randn(MC,C//G,1,1).npu(),torch.randn(MC).npu()]+bnp(MC)
                +[torch.randn(MC,1,3,3).npu(),torch.randn(MC).npu()]+bnp(MC)
                +[torch.randn(C,MC//G,1,1).npu(),torch.randn(C).npu()]+bnp(C))
    P=[torch.randn(C,IC,3,3).npu(),torch.randn(C).npu()]+bnp(C)+sup(C,MC)+sup(C,MC)
    P+=[torch.randn(C5,C,1,1).npu(),torch.randn(C5).npu()]+bnp(C5)
    P+=[torch.randn(NC,C5).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def bnr(r,o):return F.batch_norm(r,cp[o],cp[o+1],cp[o+2],cp[o+3],eps=e)
    def sur(r,o):
        BS_,C_,H_,W_=r.shape;MC_=cp[o].shape[0]
        h=F.conv2d(r,cp[o],cp[o+1],groups=G);h=F.relu(bnr(h,o+2))
        h=h.view(BS_,G,-1,H_,W_).transpose(1,2).contiguous().view(BS_,MC_,H_,W_)
        h=F.conv2d(F.pad(h,(1,1,1,1)),cp[o+6],bias=cp[o+7],groups=MC_);h=bnr(h,o+8)
        h=F.conv2d(h,cp[o+12],cp[o+13],groups=G);return bnr(h,o+14)
    r=F.conv2d(F.pad(xc,(1,1,1,1)),cp[0],cp[1]);r=F.relu(bnr(r,2))
    r=F.max_pool2d(r,2,2)
    r=sur(r,6);r=sur(r,24)
    r=F.conv2d(r,cp[42],cp[43]);r=F.relu(bnr(r,44))
    r=r.mean(dim=[2,3]);r=F.linear(r,cp[48],cp[49])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_026_shufflenet passed")
