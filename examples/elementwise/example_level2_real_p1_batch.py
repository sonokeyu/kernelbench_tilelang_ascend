"""P1 real fused materialized epilogues for L2 #16/#58/#74/#88/#91.

Each entry is a separate static JIT function. Inputs are arbitrary materialized
outputs at the stated graph boundary; upstream Mish/Softmax/GroupNorm/LogSumExp
stages are deliberately outside the local fusion boundary where applicable.
"""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC = {tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
      tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
      tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


def _setup(M, N, bm, bn):
    return T.ceildiv(M, bm), T.ceildiv(N, bn), bm // 2


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_016_add_hardtanh_scale(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = _setup(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn; rs, cs = bm * block_M + vid * sub, bn * block_N
            x=T.alloc_shared((sub,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.add(x,x,0.5); T.tile.max(x,x,-1.0); T.tile.min(x,x,1.0); T.tile.mul(x,x,2.0); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_074_leaky_mul_leaky(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=_setup(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype), RowMul:T.Tensor((M,),dtype), Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N
            x=T.alloc_shared((sub,block_N),dtype); bc=T.alloc_shared((sub,block_N),dtype); one=T.alloc_shared((sub,),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.leaky_relu(x,x,0.2); T.copy(RowMul[rs:rs+sub],one); T.tile.broadcast(bc,one); T.tile.mul(x,x,bc); T.tile.leaky_relu(x,x,0.2); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_088_swish_mul_swish(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=_setup(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype), ColMul:T.Tensor((N,),dtype), Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N
            x=T.alloc_shared((sub,block_N),dtype); t=T.alloc_shared((sub,block_N),dtype); col=T.alloc_shared((1,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.sigmoid(t,x); T.tile.mul(x,x,t)
            T.copy(ColMul[cs:cs+block_N],col)
            for r in T.serial(sub): T.copy(col,t[r:r+1,0:block_N])
            T.tile.mul(x,x,t); T.tile.sigmoid(t,x); T.tile.mul(x,x,t); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_091_bias_scale_sigmoid(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn,sub=_setup(M,N,block_M,block_N)
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype), RowBias:T.Tensor((M,),dtype), Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N
            x=T.alloc_shared((sub,block_N),dtype); bc=T.alloc_shared((sub,block_N),dtype); one=T.alloc_shared((sub,),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.copy(RowBias[rs:rs+sub],one); T.tile.broadcast(bc,one); T.tile.add(x,x,bc); T.tile.mul(x,x,2.0); T.tile.sigmoid(x,x); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main


if __name__ == "__main__":
    torch.manual_seed(0); M,N=64,2048; x=torch.randn(M,N,dtype=torch.float32).npu(); rb=torch.randn(M,dtype=torch.float32).npu(); cm=torch.randn(N,dtype=torch.float32).npu(); sb=torch.randn(1,dtype=torch.float32).npu()
    cases=[(level2_016_add_hardtanh_scale,(x,),torch.clamp(x+.5,-1,1)*2),(level2_074_leaky_mul_leaky,(x,rb),F.leaky_relu(F.leaky_relu(x,.2)*rb[:,None],.2)),(level2_088_swish_mul_swish,(x,cm),(x*torch.sigmoid(x)*cm[None,:])*torch.sigmoid(x*torch.sigmoid(x)*cm[None,:])),(level2_091_bias_scale_sigmoid,(x,rb),torch.sigmoid((x+rb[:,None])*2))]
    for fn,args,ref in cases:
        out=fn(M,N)(*args); torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),ref.cpu(),rtol=1e-2,atol=1e-2); print(fn.__name__,"passed")
