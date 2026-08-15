"""TileLang L3 #13 DenseNet TransitionLayer. BN+ReLU+Conv1x1+AvgPool."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import bn2d,relu2d,cv,pool
S=torch.npu.synchronize;TK={'TH':8,'TW':8};EPS=1e-5

def run(x,p):
    BS=x.shape[0];cw,cb=p[:2];m,v,g,beta=p[2:6]
    inv=1.0/torch.sqrt(v+EPS)
    h=bn2d(BS,x.shape[1],x.shape[2],x.shape[3],EPS,**TK)(x,m,inv,g,beta);S()
    h=relu2d(BS,x.shape[1],h.shape[2],h.shape[3],**TK)(h);S()
    h=cv(BS,x.shape[1],cw.shape[0],h.shape[2],h.shape[3],1,1,**TK)(h,cw,cb);S()
    return pool(BS,cw.shape[0],h.shape[2],h.shape[3],2,2,**TK)(h);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,C,HW=2,64,32
    x=torch.randn(BS,C,HW,HW).npu()
    P=[torch.randn(32,C,1,1).npu(),torch.randn(32).npu(),
       torch.randn(C).npu(),torch.randn(C).abs().npu()+0.1,torch.randn(C).npu(),torch.randn(C).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();e=1e-5
    r=F.batch_norm(xc,cp[2],cp[3],cp[4],cp[5],eps=e);r=F.relu(r)
    r=F.conv2d(r,cp[0],cp[1]);r=F.max_pool2d(r,2)
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_013_transition passed")
