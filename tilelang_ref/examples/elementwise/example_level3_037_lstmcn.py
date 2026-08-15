import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import ln,ewadd2d,sigmoid2d as sg,tanh2d as th
S=torch.npu.synchronize
def run(x,h0,c0,p):
    BS=x.shape[0];HD=h0.shape[1]
    gx=ln(BS,x.shape[1],4*HD)(x,p[0],p[2]);S()
    gh=ln(BS,HD,4*HD)(h0,p[1],p[3]);S()
    gates=ewadd2d(BS,4*HD)(gx,gh);S()
    def a(fn,val):v=val.view(BS,1,HD,1).contiguous();r=fn(BS,1,HD,1)(v);S();return r.view(BS,HD)
    i=a(sg,gates[:,:HD]);f=a(sg,gates[:,HD:2*HD]);g=a(th,gates[:,2*HD:3*HD]);o=a(sg,gates[:,3*HD:])
    cn=f*c0+i*g
    return cn,cn
if __name__=="__main__":
    import torch.nn.functional as Fn
    torch.manual_seed(0);BS,IN,HD=2,32,64;x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD).npu();c0=torch.randn(BS,HD).npu()
    P=[torch.randn(4*HD,IN).npu(),torch.randn(4*HD,HD).npu(),torch.randn(4*HD).npu(),torch.randn(4*HD).npu()]
    out=run(x,h0,c0,P)
    cp=[p.cpu() for p in P]
    gt=Fn.linear(x.cpu(),cp[0],cp[2])+Fn.linear(h0.cpu(),cp[1],cp[3])
    i=torch.sigmoid(gt[:,:HD]);f=torch.sigmoid(gt[:,HD:2*HD])
    g=torch.tanh(gt[:,2*HD:3*HD]);o=torch.sigmoid(gt[:,3*HD:])
    out,cn=run(x,h0,c0,P)
    hn_ref=o*torch.tanh(f*c0.cpu()+i*g)
    cn_ref=f*c0.cpu()+i*g
    torch.testing.assert_close(out.cpu(),cn_ref,rtol=1e-2,atol=1e-2)
    torch.testing.assert_close(cn.cpu(),cn_ref,rtol=1e-2,atol=1e-2)
    print("level3_037_lstmcn passed")
