"""TileLang L3 #7 GoogleNet InceptionV1: 4 parallel conv paths + cat."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cv,cvr,pool,cat2d
S=torch.npu.synchronize;TK={'TH':4,'TW':4};P1=lambda x:F.pad(x,(1,1,1,1))

def run(x,p):
    BS=x.shape[0];P=list(p)
    # path1: 1x1 conv
    h1=cv(BS,x.shape[1],P[0].shape[0],x.shape[2],x.shape[3],1,1,**TK)(x,P[0],P[1]);S()
    # path2: 1x1 → 3x3
    h2=cvr(BS,x.shape[1],P[2].shape[0],x.shape[2],x.shape[3],1,1,**TK)(x,P[2],P[3]);S()
    h2=P1(h2);h2=cv(BS,P[2].shape[0],P[4].shape[0],h2.shape[2],h2.shape[3],3,1,**TK)(h2,P[4],P[5]);S()
    # path3: 1x1 → 3x3
    h3=cvr(BS,x.shape[1],P[6].shape[0],x.shape[2],x.shape[3],1,1,**TK)(x,P[6],P[7]);S()
    h3=P1(h3);h3=cv(BS,P[6].shape[0],P[8].shape[0],h3.shape[2],h3.shape[3],3,1,**TK)(h3,P[8],P[9]);S()
    # path4: maxpool → 1x1
    h4=pool(BS,x.shape[1],x.shape[2],x.shape[3],3,1,**TK)(x);S()
    h4=cv(BS,x.shape[1],P[10].shape[0],h4.shape[2],h4.shape[3],1,1,**TK)(h4,P[10],P[11]);S()
    # cat all paths on channel dim
    h=cat2d(BS,h1.shape[1],h2.shape[1],h1.shape[2],h1.shape[3],**TK)(h1,h2);S()
    h=cat2d(BS,h.shape[1],h3.shape[1],h.shape[2],h.shape[3],**TK)(h,h3);S()
    return cat2d(BS,h.shape[1],h4.shape[1],h.shape[2],h.shape[3],**TK)(h,h4);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,IC,HW=2,3,16;O1,O2,O3,O4=8,16,8,4
    x=torch.randn(BS,IC,HW,HW).npu()
    P=[torch.randn(O1,IC,1,1).npu(),torch.randn(O1).npu(),
       torch.randn(O2,IC,1,1).npu(),torch.randn(O2).npu(),torch.randn(O2,O2,3,3).npu(),torch.randn(O2).npu(),
       torch.randn(O3,IC,1,1).npu(),torch.randn(O3).npu(),torch.randn(O3,O3,3,3).npu(),torch.randn(O3).npu(),
       torch.randn(O4,IC,1,1).npu(),torch.randn(O4).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    h1=F.conv2d(xc,cp[0],cp[1])
    h2=F.relu(F.conv2d(xc,cp[2],cp[3]));h2=F.conv2d(F.pad(h2,(1,1,1,1)),cp[4],cp[5])
    h3=F.relu(F.conv2d(xc,cp[6],cp[7]));h3=F.conv2d(F.pad(h3,(1,1,1,1)),cp[8],cp[9])
    h4=F.max_pool2d(xc,3,1,padding=1);h4=F.conv2d(h4,cp[10],cp[11])
    r=torch.cat([h1,h2,h3,h4],1)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_007_inceptionv1 passed")
