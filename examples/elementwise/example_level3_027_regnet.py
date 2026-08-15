"""TileLang L3 #27 RegNet: stage(conv+bn+relu x2 + maxpool) x2 + GAP + fc."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,pool,gap2d,ln
S=torch.npu.synchronize;EPS=1e-5;P1=lambda t:F.pad(t,(1,1,1,1))
TK={'TH':4,'TW':4}

def stage(x,ic,oc,w1,b1,m1,v1,g1,bt1,w2,b2,m2,v2,g2,bt2):
    BS=x.shape[0]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=P1(x);h=cv(BS,ic,oc,h.shape[2],h.shape[3],3,1,**TK)(h,w1,b1);S()
    h=bn2d(BS,oc,h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,bt1);S()
    h=relu2d(BS,oc,h.shape[2],h.shape[3],**TK)(h);S()
    h=P1(h);h=cv(BS,oc,oc,h.shape[2],h.shape[3],3,1,**TK)(h,w2,b2);S()
    h=bn2d(BS,oc,h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,bt2);S()
    h=relu2d(BS,oc,h.shape[2],h.shape[3],**TK)(h);S()
    return pool(BS,oc,h.shape[2],h.shape[3],2,2,**TK)(h)

def run(x,p):
    BS=x.shape[0]
    h=stage(x,x.shape[1],p[0].shape[0],*p[0:12]);S()
    h=stage(h,h.shape[1],p[12].shape[0],*p[12:24]);S()
    g=gap2d(BS,h.shape[1],h.shape[2],h.shape[3])(h);S()
    return ln(BS,h.shape[1],p[24].shape[0])(g,p[24],p[25])

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,C1,C2,NC=2,3,16,8,16,4
    x=torch.randn(BS,IC,HW,HW).npu()
    def sp(ic,oc):
        return [torch.randn(oc,ic,3,3).npu(),torch.randn(oc).npu(),
                torch.randn(oc).npu(),torch.randn(oc).abs().npu()+0.1,torch.randn(oc).npu(),torch.randn(oc).npu(),
                torch.randn(oc,oc,3,3).npu(),torch.randn(oc).npu(),
                torch.randn(oc).npu(),torch.randn(oc).abs().npu()+0.1,torch.randn(oc).npu(),torch.randn(oc).npu()]
    P=sp(IC,C1)+sp(C1,C2)+[torch.randn(NC,C2).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def st(r,o):
        r=F.conv2d(F.pad(r,(1,1,1,1)),cp[o],cp[o+1])
        r=F.batch_norm(r,cp[o+2],cp[o+3],cp[o+4],cp[o+5],eps=e);r=F.relu(r)
        r=F.conv2d(F.pad(r,(1,1,1,1)),cp[o+6],cp[o+7])
        r=F.batch_norm(r,cp[o+8],cp[o+9],cp[o+10],cp[o+11],eps=e);r=F.relu(r)
        return F.max_pool2d(r,2,2)
    r=st(xc,0);r=st(r,12)
    r=r.mean(dim=[2,3])
    r=F.linear(r,cp[24],cp[25])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_027_regnet passed")
