"""P5 local epilogue candidates with layout-preserving boundaries.

#50 is the useful multi-stage case. #30/#96 are deliberately measured as local
post-Norm/post-pool baselines to determine whether a single stable writer stage
can beat Torch at this shape.
"""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC={tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE:True,tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:True}

def grid(M,N,bm,bn): return T.ceildiv(M,bm),T.ceildiv(N,bn),bm//2

@tilelang.jit(out_idx=[2],pass_configs=PC)
def level2_050_scale_bias_scale(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Bias:T.Tensor((M,),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype); b=T.alloc_shared((sub,block_N),dtype); one=T.alloc_shared((sub,),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.mul(x,x,0.5); T.copy(Bias[rs:rs+sub],one); T.tile.broadcast(b,one); T.tile.add(x,x,b); T.tile.mul(x,x,1.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

@tilelang.jit(out_idx=[1],pass_configs=PC)
def level2_030_post_gn_hardtanh(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype); T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.max(x,x,-2.0); T.tile.min(x,x,2.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

@tilelang.jit(out_idx=[1],pass_configs=PC)
def level2_096_post_gap_clamp(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=grid(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N; x=T.alloc_shared((sub,block_N),dtype); T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.max(x,x,0.0); T.tile.min(x,x,1.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main

if __name__=="__main__":
    torch.manual_seed(0); M,N=64,2048; x=torch.randn(M,N,dtype=torch.float32).npu(); b=torch.randn(M,dtype=torch.float32).npu()
    cases=[(level2_050_scale_bias_scale(M,N),(x,b),x*.5+b[:,None]),(level2_030_post_gn_hardtanh(M,N),(x,),torch.clamp(x,-2,2)),(level2_096_post_gap_clamp(M,N),(x,),torch.clamp(x,0,1))]
    for fn,args,ref in cases:
        out=fn(*args); torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),ref.cpu(),rtol=1e-2,atol=1e-2); print("passed")
