"""TileLang L3 #18 SqueezeNet (simplified: FireModule → FireModule → FireModule → conv1x1)."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cvr,cat2d,cv
S=torch.npu.synchronize;TK={'TH':4,'TW':4}
def fire(x,iw,ib,ew1,eb1,ew3,eb3):
    BS=x.shape[0];h=cvr(BS,x.shape[1],iw.shape[0],x.shape[2],x.shape[3],1,1,**TK)(x,iw,ib);S()
    e1=cvr(BS,iw.shape[0],ew1.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,ew1,eb1);S()
    e3=F.pad(h,(1,1,1,1));e3=cvr(BS,iw.shape[0],ew3.shape[0],e3.shape[2],e3.shape[3],3,1,**TK)(e3,ew3,eb3);S()
    return cat2d(BS,ew1.shape[0],ew3.shape[0],e1.shape[2],e1.shape[3],**TK)(e1,e3)
def run(x,p):
    BS=x.shape[0];P=list(p)
    # Fire1: 3→64 squeeze, 64+64 expand
    h=fire(x,P[0],P[1],P[2],P[3],P[4],P[5]);S()
    # Fire2: 128→64 squeeze, 64+64 expand
    h=fire(h,P[6],P[7],P[8],P[9],P[10],P[11]);S()
    # Fire3: 128→64 squeeze, 64+64 expand
    h=fire(h,P[12],P[13],P[14],P[15],P[16],P[17]);S()
    # Final 1x1 conv
    h=cvr(BS,h.shape[1],P[18].shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,P[18],P[19]);S()
    return h

if __name__=="__main__":
    torch.manual_seed(0);BS,HW=2,32
    x=torch.randn(BS,3,HW,HW).npu()
    S1,E=16,32  # squeeze1=16, expand=32 each
    P=[torch.randn(S1,3,1,1).npu(),torch.randn(S1).npu(),
       torch.randn(E,S1,1,1).npu(),torch.randn(E).npu(),
       torch.randn(E,S1,3,3).npu(),torch.randn(E).npu(),
       # Fire2: input=64 (E+E), sq=16
       torch.randn(S1,2*E,1,1).npu(),torch.randn(S1).npu(),
       torch.randn(E,S1,1,1).npu(),torch.randn(E).npu(),
       torch.randn(E,S1,3,3).npu(),torch.randn(E).npu(),
       # Fire3: input=64
       torch.randn(S1,2*E,1,1).npu(),torch.randn(S1).npu(),
       torch.randn(E,S1,1,1).npu(),torch.randn(E).npu(),
       torch.randn(E,S1,3,3).npu(),torch.randn(E).npu(),
       # Final: 64→64
       torch.randn(64,2*E,1,1).npu(),torch.randn(64).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    r=F.relu(F.conv2d(xc,cp[0],cp[1]))
    r=torch.cat([F.relu(F.conv2d(r,cp[2],cp[3])),F.relu(F.conv2d(r,cp[4],cp[5],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[6],cp[7]))
    r=torch.cat([F.relu(F.conv2d(r,cp[8],cp[9])),F.relu(F.conv2d(r,cp[10],cp[11],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[12],cp[13]))
    r=torch.cat([F.relu(F.conv2d(r,cp[14],cp[15])),F.relu(F.conv2d(r,cp[16],cp[17],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[18],cp[19]))
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    r=F.relu(F.conv2d(xc,cp[0],cp[1]))
    r=torch.cat([F.relu(F.conv2d(r,cp[2],cp[3])),F.relu(F.conv2d(r,cp[4],cp[5],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[6],cp[7]))
    r=torch.cat([F.relu(F.conv2d(r,cp[8],cp[9])),F.relu(F.conv2d(r,cp[10],cp[11],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[12],cp[13]))
    r=torch.cat([F.relu(F.conv2d(r,cp[14],cp[15])),F.relu(F.conv2d(r,cp[16],cp[17],padding=1))],1)
    r=F.relu(F.conv2d(r,cp[18],cp[19]))
    torch.testing.assert_close(out.cpu(),r,rtol=5e-2,atol=3e-2)
    print("level3_018_squeezenet passed")
