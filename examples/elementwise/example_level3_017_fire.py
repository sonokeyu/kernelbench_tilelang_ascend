"""TileLang L3 #17 SqueezeNet FireModule."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cvr,cat2d
S=torch.npu.synchronize

def run(x,p):
    BS=x.shape[0];sq_w,sq_b,e1w,e1b,e3w,e3b=p[:6]
    t={'TH':4,'TW':4}
    # squeeze: 1x1 conv+relu
    h=cvr(BS,x.shape[1],sq_w.shape[0],x.shape[2],x.shape[3],1,1,**t)(x,sq_w,sq_b);S()
    # expand1x1 + expand3x3 parallel, then cat
    e1=cvr(BS,sq_w.shape[0],e1w.shape[0],h.shape[2],h.shape[3],1,1,**t)(h,e1w,e1b);S()
    e3=F.pad(h,(1,1,1,1))
    e3=cvr(BS,sq_w.shape[0],e3w.shape[0],e3.shape[2],e3.shape[3],3,1,**t)(e3,e3w,e3b);S()
    return cat2d(BS,e1w.shape[0],e3w.shape[0],e1.shape[2],e1.shape[3],**t)(e1,e3)

if __name__=="__main__":
    torch.manual_seed(0)
    BS,IC,HW=2,3,32;SQ,EX1,EX3=6,32,32
    x=torch.randn(BS,IC,HW,HW).npu()
    P=[torch.randn(SQ,IC,1,1).npu(),torch.randn(SQ).npu(),
       torch.randn(EX1,SQ,1,1).npu(),torch.randn(EX1).npu(),
       torch.randn(EX3,SQ,3,3).npu(),torch.randn(EX3).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    r=F.relu(F.conv2d(xc,cp[0],cp[1]))
    r=torch.cat([F.relu(F.conv2d(r,cp[2],cp[3])),
                 F.relu(F.conv2d(r,cp[4],cp[5],padding=1))],1)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_017_fire passed")
