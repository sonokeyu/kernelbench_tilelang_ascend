"""TileLang L3 #45 UNetSoftmax: DoubleConv enc/dec + ConvTranspose upsample + softmax."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,bn2d,relu2d,pool,convT2x2,cat2d
S=torch.npu.synchronize;EPS=1e-5;TK={'TH':4,'TW':4};P1=lambda t:F.pad(t,(1,1,1,1))

def dconv(x,p,o):
    """conv3x3+bn+relu x2 (DoubleConv)."""
    BS=x.shape[0];C=x.shape[1]
    w1,b1,m1,v1,g1,t1,w2,b2,m2,v2,g2,t2=p[o:o+12]
    i1=1.0/torch.sqrt(v1+EPS);i2=1.0/torch.sqrt(v2+EPS)
    h=P1(x);h=cv(BS,C,w1.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,w1,b1);S()
    h=bn2d(BS,w1.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m1,i1,g1,t1);S()
    h=relu2d(BS,w1.shape[0],h.shape[2],h.shape[3],**TK)(h);S()
    h=P1(h);h=cv(BS,w1.shape[0],w2.shape[0],h.shape[2],h.shape[3],3,1,**TK)(h,w2,b2);S()
    h=bn2d(BS,w2.shape[0],h.shape[2],h.shape[3],EPS,**TK)(h,m2,i2,g2,t2);S()
    return relu2d(BS,w2.shape[0],h.shape[2],h.shape[3],**TK)(h)

def run(x,p):
    BS=x.shape[0]
    e1=dconv(x,p,0);S()
    pl=pool(BS,e1.shape[1],e1.shape[2],e1.shape[3],2,2,**TK)(e1);S()
    bn=dconv(pl,p,12);S()
    up=convT2x2(BS,bn.shape[1],e1.shape[1],bn.shape[2],bn.shape[3],**TK)(bn,p[24],p[25]);S()
    cc=cat2d(BS,up.shape[1],e1.shape[1],up.shape[2],up.shape[3],**TK)(up,e1);S()
    d1=dconv(cc,p,26);S()
    fw,fb=p[38],p[39]
    out=cv(BS,d1.shape[1],fw.shape[0],d1.shape[2],d1.shape[3],1,1,**TK)(d1,fw,fb);S()
    return F.softmax(out,dim=1)

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW,FT,NC=2,3,16,4,2
    x=torch.randn(BS,IC,HW,HW).npu()
    def bnp(c):return [torch.randn(c).npu(),torch.randn(c).abs().npu()+0.1,torch.randn(c).npu(),torch.randn(c).npu()]
    def dcp(ic,oc):
        return ([torch.randn(oc,ic,3,3).npu(),torch.randn(oc).npu()]+bnp(oc)
                +[torch.randn(oc,oc,3,3).npu(),torch.randn(oc).npu()]+bnp(oc))
    P=dcp(IC,FT)+dcp(FT,FT*2)
    P+=[torch.randn(FT*2,FT,2,2).npu(),torch.randn(FT).npu()]
    P+=dcp(FT*2,FT)
    P+=[torch.randn(NC,FT,1,1).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    def bnr(r,o):return F.batch_norm(r,cp[o],cp[o+1],cp[o+2],cp[o+3],eps=e)
    def dcr(r,o):
        r=F.conv2d(F.pad(r,(1,1,1,1)),cp[o],cp[o+1]);r=F.relu(bnr(r,o+2))
        r=F.conv2d(F.pad(r,(1,1,1,1)),cp[o+6],cp[o+7]);return F.relu(bnr(r,o+8))
    e1=dcr(xc,0)
    pl=F.max_pool2d(e1,2,2)
    bn=dcr(pl,12)
    up=F.conv_transpose2d(bn,cp[24],cp[25],stride=2)
    cc=torch.cat([up,e1],1)
    d1=dcr(cc,26)
    o=F.conv2d(d1,cp[38],cp[39])
    ref=F.softmax(o,dim=1)
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-2,atol=1e-2)
    print("level3_045_unet_softmax passed")
