"""TileLang L3 #38 Bidirectional LSTM."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as Fn
from _l3_kernels import ln,ewadd2d,sigmoid2d as sg,tanh2d as th
S=torch.npu.synchronize
def cell(x,h,c,wi,wh,bi,bh):
    BS,HD=x.shape[0],h.shape[1]
    gx=ln(BS,x.shape[1],4*HD)(x,wi,bi);S()
    gh=ln(BS,HD,4*HD)(h,wh,bh);S()
    g=ewadd2d(BS,4*HD)(gx,gh);S()
    def a(f,v):v=v.view(BS,1,HD,1).contiguous();r=f(BS,1,HD,1)(v);S();return r.view(BS,HD)
    i=a(sg,g[:,:HD]);f=a(sg,g[:,HD:2*HD]);g2=a(th,g[:,2*HD:3*HD]);o=a(sg,g[:,3*HD:])
    cn=f*c+i*g2;return o*torch.tanh(cn),cn
def run(x,h0,c0,p):
    HD=h0.shape[1]//2
    hf,_=cell(x,h0[:,:HD].contiguous(),c0[:,:HD].contiguous(),p[0],p[1],p[2],p[3])
    xr=torch.flip(x,[0]).contiguous()
    hr,_=cell(xr,h0[:,HD:].contiguous(),c0[:,HD:].contiguous(),p[4],p[5],p[6],p[7])
    return torch.cat([hf,torch.flip(hr,[0])],1)
if __name__=="__main__":
    BS,IN,HD=2,32,32
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD*2).npu();c0=torch.randn(BS,HD*2).npu()
    P=[torch.randn(4*HD,IN).npu(),torch.randn(4*HD,HD).npu(),torch.randn(4*HD).npu(),torch.randn(4*HD).npu(),
       torch.randn(4*HD,IN).npu(),torch.randn(4*HD,HD).npu(),torch.randn(4*HD).npu(),torch.randn(4*HD).npu()]
    out=run(x,h0,c0,P)
    cp=[p.cpu() for p in P]
    def cc(xi,hi,ci,wi,wh,bi,bh):
        g=Fn.linear(xi,wi,bi)+Fn.linear(hi,wh,bh)
        i=torch.sigmoid(g[:,:HD]);f=torch.sigmoid(g[:,HD:2*HD])
        g2=torch.tanh(g[:,2*HD:3*HD]);o=torch.sigmoid(g[:,3*HD:])
        return o*torch.tanh(f*ci+i*g2),f*ci+i*g2
    hf,_=cc(x.cpu(),h0[:,:HD].cpu(),c0[:,:HD].cpu(),cp[0],cp[1],cp[2],cp[3])
    hr,_=cc(torch.flip(x.cpu(),[0]),h0[:,HD:].cpu(),c0[:,HD:].cpu(),cp[4],cp[5],cp[6],cp[7])
    ref=torch.cat([hf,torch.flip(hr,[0])],1)
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-2,atol=1e-2)
    print("level3_038_bilstm passed")
