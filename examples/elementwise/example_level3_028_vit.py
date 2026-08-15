"""TileLang L3 #28 ViT: tilelang MHA+MLP linear, torch softmax/LN. Pre-embedded input."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,math
from _l3_kernels import ln,lr
S=torch.npu.synchronize

def run(x,p):
    BS,L,C=x.shape;NH=2;HS=C//NH
    qw,kw,vw,qb,kb,vb,ow,ob,ln1w,ln1b,fc1w,fc1b,fc2w,fc2b,ln2w,ln2b,outw,outb=p
    # single transformer block
    BL=BS*L
    q=ln(BL,C,C)(x.reshape(BL,C).contiguous(),qw,qb);S()
    k=ln(BL,C,C)(x.reshape(BL,C).contiguous(),kw,kb);S()
    v=ln(BL,C,C)(x.reshape(BL,C).contiguous(),vw,vb);S()
    q=q.view(BS,L,NH,HS).transpose(1,2);k=k.view(BS,L,NH,HS).transpose(1,2)
    v=v.view(BS,L,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS);att=torch.softmax(att,-1)
    y=att@v;y=y.transpose(1,2).contiguous().view(BL,C)
    a=ln(BL,C,C)(y,ow,ob);S()
    xl=torch.nn.functional.layer_norm(a.view(BS,L,C)+x,(C,),ln1w,ln1b)
    m=lr(BL,C,fc1w.shape[0])(xl.reshape(BL,C).contiguous(),fc1w,fc1b);S()
    m=torch.nn.functional.gelu(m)
    m=ln(BL,fc1w.shape[0],C)(m.reshape(BL,fc1w.shape[0]).contiguous(),fc2w,fc2b);S()
    x3=xl+m.view(BS,L,C)
    x3l=torch.nn.functional.layer_norm(x3,(C,),ln2w,ln2b)
    return ln(BS,C,outw.shape[0])(x3l[:,0,:].contiguous(),outw,outb);S()

if __name__=="__main__":
    torch.manual_seed(0);BS,L,C,NC=2,4,8,4;NH=2;HS=C//NH
    x=torch.randn(BS,L,C).npu()
    P=[torch.randn(C,C).npu(),torch.randn(C,C).npu(),torch.randn(C,C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(C,C).npu(),torch.randn(C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(C*2,C).npu(),torch.randn(C*2).npu(),
       torch.randn(C,C*2).npu(),torch.randn(C).npu(),
       torch.randn(C).npu(),torch.randn(C).npu(),
       torch.randn(NC,C).npu(),torch.randn(NC).npu()]
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu();BL=BS*L
    q=torch.nn.functional.linear(xc.reshape(BL,C),cp[0],cp[3]).view(BS,L,NH,HS).transpose(1,2)
    k=torch.nn.functional.linear(xc.reshape(BL,C),cp[1],cp[4]).view(BS,L,NH,HS).transpose(1,2)
    v=torch.nn.functional.linear(xc.reshape(BL,C),cp[2],cp[5]).view(BS,L,NH,HS).transpose(1,2)
    att=(q@k.transpose(-2,-1))/math.sqrt(HS);att=torch.softmax(att,-1)
    y=att@v;y=y.transpose(1,2).contiguous().view(BL,C)
    a=torch.nn.functional.linear(y,cp[6],cp[7]).view(BS,L,C)
    xl=torch.nn.functional.layer_norm(a+xc,(C,),cp[8],cp[9])
    m=torch.nn.functional.gelu(torch.nn.functional.linear(xl.reshape(BL,C),cp[10],cp[11]))
    m=torch.nn.functional.linear(m,cp[12],cp[13]).view(BS,L,C)
    x3=xl+m
    x3l=torch.nn.functional.layer_norm(x3,(C,),cp[14],cp[15])
    ref=torch.nn.functional.linear(x3l[:,0,:],cp[16],cp[17])
    torch.testing.assert_close(out.cpu(),ref,rtol=5e-2,atol=5e-2)
    print("level3_028_vit passed")
