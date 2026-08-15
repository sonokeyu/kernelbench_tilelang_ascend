"""TileLang L3 #41 Bidirectional GRU."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as Fn
from _l3_kernels import ln,ewadd2d,sigmoid2d as sg,tanh2d as th
S=torch.npu.synchronize
def run(x,h0,p):
    BS,HD=x.shape[0],h0.shape[1]//2;TD=3*HD
    def c(xi,hi,wi,wh,bi,bh):
        g=ewadd2d(BS,TD)(ln(BS,xi.shape[1],TD)(xi,wi,bi),ln(BS,HD,TD)(hi,wh,bh));S()
        def a(f,v):v=v.view(BS,1,HD,1).contiguous();r=f(BS,1,HD,1)(v);S();return r.view(BS,HD)
        return (1-a(sg,g[:,:HD]))*a(th,g[:,2*HD:])+a(sg,g[:,:HD])*hi
    hf=c(x,h0[:,:HD].contiguous(),p[0],p[1],p[2],p[3])
    xr=torch.flip(x,[0]).contiguous()
    hr=c(xr,h0[:,HD:].contiguous(),p[4],p[5],p[6],p[7])
    return torch.cat([hf,torch.flip(hr,[0])],1)
if __name__=="__main__":
    BS,IN,HD=2,32,32
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD*2).npu()
    P=[torch.randn(3*HD,IN).npu(),torch.randn(3*HD,HD).npu(),torch.randn(3*HD).npu(),torch.randn(3*HD).npu(),
       torch.randn(3*HD,IN).npu(),torch.randn(3*HD,HD).npu(),torch.randn(3*HD).npu(),torch.randn(3*HD).npu()]
    out=run(x,h0,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    def gr(xi,hi,wi,wh,bi,bh):
        g=Fn.linear(xi,wi,bi)+Fn.linear(hi,wh,bh)
        z=torch.sigmoid(g[:,:HD]);r=torch.sigmoid(g[:,HD:2*HD])
        n=torch.tanh(g[:,2*HD:])
        return (1-z)*n+z*hi
    hf=gr(xc,h0[:,:HD].cpu(),cp[0],cp[1],cp[2],cp[3])
    hr=gr(torch.flip(xc,[0]),h0[:,HD:].cpu(),cp[4],cp[5],cp[6],cp[7])
    ref=torch.cat([hf,torch.flip(hr,[0])],1)
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-2,atol=1e-2)
    print("level3_041_bigru passed")
