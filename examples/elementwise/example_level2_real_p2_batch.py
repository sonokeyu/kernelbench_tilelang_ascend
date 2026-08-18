"""P2 local real-fusion candidates with explicit materialized boundaries.

#53: GEMM output -> scale -> HardTanh; GELU remains outside.
#13: post-channel-Softmax output -> Tanh -> scale; Conv/mean/Softmax remain outside.
#89: post-channel-Softmax output -> channel subtract -> Swish; Max remains outside.
For Conv3d cases, flatten (B,C,1,H,W) to M=B*C,N=H*W, so channel parameters
are row-wise and the broadcast axis is explicit.
"""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC={tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE:True,tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:True}

def grid(M,N,bm,bn): return T.ceildiv(M,bm),T.ceildiv(N,bn),bm//2

@tilelang.jit(out_idx=[1],pass_configs=PC)
def level2_053_scale_hardtanh(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.mul(x,x,0.5); T.tile.max(x,x,-2.0); T.tile.min(x,x,2.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

@tilelang.jit(out_idx=[1],pass_configs=PC)
def level2_013_postsoftmax_tanh_scale(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn
            rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.tanh(x,x); T.tile.mul(x,x,2.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

@tilelang.jit(out_idx=[2],pass_configs=PC)
def level2_089_postsoftmax_sub_swish(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),ChannelSub:T.Tensor((M,),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn
            rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype); t=T.alloc_shared((sub,block_N),dtype); one=T.alloc_shared((sub,),dtype); sub2d=T.alloc_shared((sub,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.copy(ChannelSub[rs:rs+sub],one); T.tile.broadcast(sub2d,one); T.tile.sub(x,x,sub2d); T.tile.sigmoid(t,x); T.tile.mul(x,x,t); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

if __name__=="__main__":
    torch.manual_seed(0); M,N=64,2048; x=torch.randn(M,N,dtype=torch.float32).npu(); sub=torch.randn(M,dtype=torch.float32).npu()
    cases=[(level2_053_scale_hardtanh(M,N),(x,),F.hardtanh(x*.5,-2,2)),(level2_013_postsoftmax_tanh_scale(M,N),(x,),torch.tanh(x)*2),(level2_089_postsoftmax_sub_swish(M,N),(x,sub),(x-sub[:,None])*torch.sigmoid(x-sub[:,None]))]
    for fn,args,ref in cases:
        out=fn(*args); torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),ref.cpu(),rtol=1e-2,atol=1e-2); print("passed")
