"""TileLang L3 #30 SwinTransformerV2 WindowAttention: cosine attn + logit_scale + rel-pos-bias."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import ln,sigmoid2d
S=torch.npu.synchronize

def tl_sigmoid(t):
    sh=t.shape;n=t.numel()
    r=sigmoid2d(n,1,1,1)(t.reshape(n,1,1,1).contiguous());S()
    return r.reshape(sh)

def run(x,p,num_heads):
    """x: (B_, N, C) window features."""
    B_,N,C=x.shape;NH=num_heads;HD=C//NH
    qkv_w,qkv_b,logit_scale,rpb,proj_w,proj_b=p
    # qkv linear (tilelang)
    qkv=ln(B_*N,C,3*C)(x.reshape(B_*N,C).contiguous(),qkv_w,qkv_b);S()
    qkv=qkv.reshape(B_,N,3,NH,HD).permute(2,0,3,1,4)
    q,k,v=qkv[0],qkv[1],qkv[2]
    # cosine attention
    attn=F.normalize(q,dim=-1)@F.normalize(k,dim=-1).transpose(-2,-1)
    ls=torch.clamp(logit_scale,max=torch.log(torch.tensor(100.0,device=x.device))).exp()
    attn=attn*ls
    # relative position bias: 16*sigmoid (tilelang sigmoid)
    rpb_s=16*tl_sigmoid(rpb)
    attn=attn+rpb_s.unsqueeze(0)
    attn=F.softmax(attn,dim=-1)
    out=(attn@v).transpose(1,2).reshape(B_*N,C)
    return ln(B_*N,C,C)(out.contiguous(),proj_w,proj_b).view(B_,N,C)

if __name__=="__main__":
    torch.manual_seed(0);B_,N,C,NH=2,4,8,2;HD=C//NH
    x=torch.randn(B_,N,C).npu()
    P=[torch.randn(3*C,C).npu(),torch.randn(3*C).npu(),
       torch.randn(NH,1,1).npu()*0.1,torch.randn(NH,N,N).npu(),
       torch.randn(C,C).npu(),torch.randn(C).npu()]
    out=run(x,P,NH)
    cp=[t.cpu() for t in P];xc=x.cpu()
    qkv=F.linear(xc.reshape(B_*N,C),cp[0],cp[1]).reshape(B_,N,3,NH,HD).permute(2,0,3,1,4)
    q,k,v=qkv[0],qkv[1],qkv[2]
    attn=F.normalize(q,dim=-1)@F.normalize(k,dim=-1).transpose(-2,-1)
    ls=torch.clamp(cp[2],max=torch.log(torch.tensor(100.0))).exp()
    attn=attn*ls
    attn=attn+(16*torch.sigmoid(cp[3])).unsqueeze(0)
    attn=F.softmax(attn,dim=-1)
    o=(attn@v).transpose(1,2).reshape(B_*N,C)
    ref=F.linear(o,cp[4],cp[5]).view(B_,N,C)
    torch.testing.assert_close(out.cpu(),ref,rtol=5e-2,atol=5e-2)
    print("level3_030_swinv2_window_attn passed")
