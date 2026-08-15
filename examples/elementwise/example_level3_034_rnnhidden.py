"""TileLang L3 #34 VanillaRNNHidden. Same as #33 but returns output+hidden."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import ln,tanh2d
S=torch.npu.synchronize

def run(x,h0,p):
    BS,IN,HD=x.shape[0],x.shape[1],h0.shape[1]
    xh=torch.cat([x,h0],1).contiguous()
    h1=ln(BS,IN+HD,HD)(xh,p[0],p[1]);S()
    h1_4d=h1.view(BS,1,HD,1).contiguous()
    h2_4d=tanh2d(BS,1,HD,1)(h1_4d);S()
    h2=h2_4d.view(BS,HD).contiguous()
    out=ln(BS,HD,p[2].shape[0])(h2,p[2],p[3]);S()
    return out,h2

if __name__=="__main__":
    torch.manual_seed(0);BS,IN,HD,OUT=2,32,64,16
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD).npu()
    P=[torch.randn(HD,IN+HD).npu(),torch.randn(HD).npu(),
       torch.randn(OUT,HD).npu(),torch.randn(OUT).npu()]
    out,h=run(x,h0,P)
    cp=[p.cpu() for p in P];xc=x.cpu();hc=h0.cpu()
    xh=torch.cat([xc,hc],1)
    h_ref=torch.tanh(torch.nn.functional.linear(xh,cp[0],cp[1]))
    r=torch.nn.functional.linear(h_ref,cp[2],cp[3])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    torch.testing.assert_close(h.cpu(),h_ref,rtol=1e-2,atol=1e-2)
    print("level3_034_rnnhidden passed")
