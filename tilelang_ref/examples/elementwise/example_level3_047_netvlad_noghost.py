"""TileLang L3 #47 NetVLAD no ghost clusters: tilelang matmul, torch BN/softmax/normalize."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import ln
S=torch.npu.synchronize

def run(x,p,cluster_size,ghost=1):
    """x: (B,N,D). clusters:(D,K+G), clusters2:(1,D,K), bn params."""
    B,N,D=x.shape
    clusters,clusters2,bn_m,bn_v,bn_g,bn_b=p
    KG=clusters.shape[1];eps=1e-5
    x2=x.reshape(B*N,D).contiguous()
    # assignment = x @ clusters  (tilelang matmul, zero bias)
    zb=torch.zeros(KG,dtype=x.dtype).npu()
    a=ln(B*N,D,KG)(x2,clusters.t().contiguous(),zb);S()
    # BatchNorm1d + softmax
    a=F.batch_norm(a,bn_m,bn_v,bn_g,bn_b,eps=eps)
    a=F.softmax(a,dim=1)[:,:cluster_size]
    a=a.view(B,N,cluster_size)
    a_sum=a.sum(dim=1,keepdim=True)
    ac=a_sum*clusters2
    at=a.transpose(1,2)
    xv=x2.view(B,N,D)
    vlad=torch.matmul(at,xv).transpose(1,2)-ac
    vlad=F.normalize(vlad)
    return F.normalize(vlad.reshape(B,-1))

if __name__=="__main__":
    torch.manual_seed(0);B,N,D,K,G=2,8,16,4,0
    x=torch.randn(B,N,D).npu()
    P=[torch.randn(D,K+G).npu(),torch.randn(1,D,K).npu(),
       torch.randn(K+G).npu(),torch.randn(K+G).abs().npu()+0.1,
       torch.randn(K+G).npu(),torch.randn(K+G).npu()]
    out=run(x,P,K,G)
    cp=[t.cpu() for t in P];xc=x.cpu();e=1e-5
    x2=xc.reshape(B*N,D)
    a=torch.matmul(x2,cp[0])
    a=F.batch_norm(a,cp[2],cp[3],cp[4],cp[5],eps=e)
    a=F.softmax(a,dim=1)[:,:K].view(B,N,K)
    ac=a.sum(dim=1,keepdim=True)*cp[1]
    vlad=torch.matmul(a.transpose(1,2),x2.view(B,N,D)).transpose(1,2)-ac
    vlad=F.normalize(vlad)
    ref=F.normalize(vlad.reshape(B,-1))
    torch.testing.assert_close(out.cpu(),ref,rtol=5e-2,atol=5e-2)
    print("level3_046_netvlad_noghost passed")
