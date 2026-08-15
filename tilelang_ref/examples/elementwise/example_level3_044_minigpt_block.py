"""TileLang L3 #44 MiniGPTBlock: tilelang attn/MLP linear, torch LN/softmax/gelu."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,math
from _l3_kernels import ln,lr
S=torch.npu.synchronize

def run(x,p):
    BS,T,C=x.shape;NH=2;HS=C//NH;BL=BS*T
    qw,kw,vw,qb,kb,vb,ow,ob,ln1w,ln1b,fc1w,fc1b,fc2w,fc2b,ln2w,ln2b=p
    # LN1
    x_ln=torch.nn.functional.layer_norm(x,(C,),ln1w,ln1b)
    # MHA
    q=ln(BL,C,C)(x_ln.reshape(BL,C).contiguous(),qw,qb);S()
    k=ln(BL,C,C)(x_ln.reshape(BL,C).contiguous(),kw,kb);S()
    v=ln(BL,C,C)(x_ln.reshape(BL,C).contiguous(),vw,vb);S()
    q=q.view(BS,T,NH,HS).transpose(1,2);k=k.view(BS,T,NH,HS).transpose(1,2)
    v=v.view(BS,T,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS)
    m=torch.tril(torch.ones(T,T)).npu()==0
    att=att.masked_fill(m.unsqueeze(0).unsqueeze(0),float('-inf'))
    att=torch.softmax(att,-1);y=att@v
    y=y.transpose(1,2).contiguous().view(BL,C)
    a=ln(BL,C,C)(y,ow,ob);S()
    x2=x+a.view(BS,T,C)
    # LN2 + MLP
    x_ln2=torch.nn.functional.layer_norm(x2,(C,),ln2w,ln2b)
    h=ln(BL,C,fc1w.shape[0])(x_ln2.reshape(BL,C).contiguous(),fc1w,fc1b);S()
    h=torch.nn.functional.gelu(h)
    h=ln(BL,fc1w.shape[0],C)(h.reshape(BL,fc1w.shape[0]).contiguous(),fc2w,fc2b);S()
    return x2+h.view(BS,T,C)

if __name__=="__main__":
    torch.manual_seed(0);BS,T,C=1,4,16;NH=2;HS=C//NH;BL=BS*T
    x=torch.randn(BS,T,C).npu()
    P=[torch.randn(C,C).npu(),torch.randn(C,C).npu(),torch.randn(C,C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(C,C).npu(),torch.randn(C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(C*4,C).npu(),torch.randn(C*4).npu(),
       torch.randn(C,C*4).npu(),torch.randn(C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    x_ln=torch.nn.functional.layer_norm(xc,(C,),cp[8],cp[9])
    q=torch.nn.functional.linear(x_ln.reshape(BL,C),cp[0],cp[3]).view(BS,T,NH,HS).transpose(1,2)
    k=torch.nn.functional.linear(x_ln.reshape(BL,C),cp[1],cp[4]).view(BS,T,NH,HS).transpose(1,2)
    v=torch.nn.functional.linear(x_ln.reshape(BL,C),cp[2],cp[5]).view(BS,T,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS)
    m=torch.tril(torch.ones(T,T))==0;att=att.masked_fill(m.unsqueeze(0).unsqueeze(0),float('-inf'))
    att=torch.softmax(att,-1);y=att@v
    y=y.transpose(1,2).contiguous().view(BL,C)
    a=torch.nn.functional.linear(y,cp[6],cp[7]).view(BS,T,C)
    ref=xc+a
    x_ln2=torch.nn.functional.layer_norm(ref,(C,),cp[14],cp[15])
    h=torch.nn.functional.gelu(torch.nn.functional.linear(x_ln2.reshape(BL,C),cp[10],cp[11]))
    h=torch.nn.functional.linear(h,cp[12],cp[13]).view(BS,T,C)
    ref=ref+h
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-1,atol=1e-1)
    print("level3_044_minigpt_block passed")
