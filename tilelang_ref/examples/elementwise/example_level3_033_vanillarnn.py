"""TileLang L3 #33 VanillaRNN: cat(x,h) → linear+tanh → hidden, linear → output."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import ln,tanh2d
S=torch.npu.synchronize

def run(x,h0,p):
    BS=x.shape[0];IN,HD=x.shape[1],h0.shape[1]
    i2hw,i2hb=p[0],p[1];h2ow,h2ob=p[2],p[3]
    # cat x and h0 along dim=1: (BS, IN+HD)
    xh=torch.cat([x,h0],dim=1).contiguous()
    # linear: i2hw @ [x;h] + i2hb → (BS, HD)
    h1=ln(BS,IN+HD,HD)(xh,i2hw,i2hb);S()
    # tanh via reshape to 4D
    h1_4d=h1.view(BS,1,HD,1).contiguous()
    h2_4d=tanh2d(BS,1,HD,1)(h1_4d);S()
    h2=h2_4d.view(BS,HD).contiguous()
    # output linear: h2ow @ h2 + h2ob → (BS, OUT)
    OUT=h2ow.shape[0]
    out=ln(BS,HD,OUT)(h2,h2ow,h2ob);S()
    return out

if __name__=="__main__":
    torch.manual_seed(0)
    BS,IN,HD,OUT=2,32,64,16
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD).npu()
    P=[torch.randn(HD,IN+HD).npu(),torch.randn(HD).npu(),
       torch.randn(OUT,HD).npu(),torch.randn(OUT).npu()]
    out=run(x,h0,P)
    cp=[p.cpu() for p in P]
    xc=x.cpu();hc=h0.cpu()
    xh=torch.cat([xc,hc],1)
    h=torch.tanh(torch.nn.functional.linear(xh,cp[0],cp[1]))
    r=torch.nn.functional.linear(h,cp[2],cp[3])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_033_vanillarnn passed")
