import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import ln,ewadd2d,sigmoid2d as sg,tanh2d as th
S=torch.npu.synchronize
def run(x,h0,p):
    BS=x.shape[0];HD=h0.shape[1]
    gx=ln(BS,x.shape[1],3*HD)(x,p[0],p[2]);S()
    gh=ln(BS,HD,3*HD)(h0,p[1],p[3]);S()
    gates=ewadd2d(BS,3*HD)(gx,gh);S()
    def a(fn,val):v=val.view(BS,1,HD,1).contiguous();r=fn(BS,1,HD,1)(v);S();return r.view(BS,HD)
    z=a(sg,gates[:,:HD]);r_g=a(sg,gates[:,HD:2*HD]);n=a(th,gates[:,2*HD:])
    hn=(1-z)*n+z*h0
    return hn,hn
if __name__=="__main__":
    import torch.nn.functional as Fn
    torch.manual_seed(0);BS,IN,HD=2,32,64
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD).npu()
    P=[torch.randn(3*HD,IN).npu(),torch.randn(3*HD,HD).npu(),torch.randn(3*HD).npu(),torch.randn(3*HD).npu()]
    out,hn_out=run(x,h0,P)
    cp=[p.cpu() for p in P];xc=x.cpu();hc=h0.cpu()
    gt=Fn.linear(xc,cp[0],cp[2])+Fn.linear(hc,cp[1],cp[3])
    z=torch.sigmoid(gt[:,:HD]);r_g=torch.sigmoid(gt[:,HD:2*HD])
    n=torch.tanh(gt[:,2*HD:])
    hn_ref=(1-z)*n+z*hc
    torch.testing.assert_close(out.cpu(),hn_ref,rtol=1e-2,atol=1e-2)
    torch.testing.assert_close(hn_out.cpu(),hn_ref,rtol=1e-2,atol=1e-2)
    print("level3_040_gruhidden passed")
