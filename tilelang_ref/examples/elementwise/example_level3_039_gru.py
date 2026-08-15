"""TileLang L3 #39 GRU: update(z)+reset(r) sigmoid, new(n) tanh."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from _l3_kernels import ln,ewadd2d,ewmul2d
S=torch.npu.synchronize

def run(x,h0,p):
    BS,IN,HD=x.shape[0],x.shape[1],h0.shape[1]
    wi,wh,bi,bh=p  # wi: (3*HD, IN), wh: (3*HD, HD)
    # Compute all gate pre-activations in one linear each
    gx=ln(BS,IN,3*HD)(x,wi,bi);S()
    gh=ln(BS,HD,3*HD)(h0,wh,bh);S()
    gates=ewadd2d(BS,3*HD)(gx,gh);S()
    # Split: z = gates[0:HD], r = gates[HD:2*HD], n = gates[2*HD:3*HD]
    z_t=gates[:,:HD];r_t=gates[:,HD:2*HD];n_t=gates[:,2*HD:]
    # Apply activations
    z_flat=z_t.contiguous().view(BS,1,HD,1)
    r_flat=r_t.contiguous().view(BS,1,HD,1)
    n_flat=n_t.contiguous().view(BS,1,HD,1)
    # Use tanh2d and sigmoid2d — need to import
    from _l3_kernels import sigmoid2d as sig,tanh2d as tanh
    z_4d=sig(BS,1,HD,1)(z_flat);S()
    r_4d=sig(BS,1,HD,1)(r_flat);S()
    n_4d=tanh(BS,1,HD,1)(n_flat);S()
    z=z_4d.view(BS,HD);r=r_4d.view(BS,HD);n=n_4d.view(BS,HD)
    # h_new = (1-z)*n + z*h0 (actually: (1-z)*h0 + z*n)
    # clamp? No, just compute
    # tilelang needs elementwise for this
    # h_new = z*h0 + (1-z)*n  → need 1-z
    # Actually h_new = (1-z)*n + z*h0
    # Let me do it in 2D with tor values: 
    # We don't have sub-scalar. Use: (1-z) = 1 + (-z) = 1 - z
    # For now, just compute with torch ops on NPU
    one=torch.ones_like(z)
    omz=one-z  # (1-z)
    z_h0=z*h0;omz_n=omz*n
    h_new=z_h0+omz_n
    return h_new

if __name__=="__main__":
    torch.manual_seed(0);BS,IN,HD=2,32,64
    x=torch.randn(BS,IN).npu();h0=torch.randn(BS,HD).npu()
    P=[torch.randn(3*HD,IN).npu(),torch.randn(3*HD,HD).npu(),
       torch.randn(3*HD).npu(),torch.randn(3*HD).npu()]
    out=run(x,h0,P)
    cp=[p.cpu() for p in P];xc=x.cpu();hc=h0.cpu()
    wi,wh,bi,bh=cp
    gx=torch.nn.functional.linear(xc,wi,bi)
    gh=torch.nn.functional.linear(hc,wh,bh)
    gates=gx+gh
    z=torch.sigmoid(gates[:,:HD]);r=torch.sigmoid(gates[:,HD:2*HD])
    n=torch.tanh(gates[:,2*HD:])
    h_new=(1-z)*n+z*hc
    torch.testing.assert_close(out.cpu(),h_new,rtol=1e-2,atol=1e-2)
    print("level3_039_gru passed")
