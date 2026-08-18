"""L2 #82 materialized Conv2d output -> tanh -> scale -> channel bias."""
import tilelang
import tilelang.language as T
import torch

PC={tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE:True,tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:True}
@tilelang.jit(out_idx=[2],pass_configs=PC)
def level2_082_tanh_scale_bias(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn=T.ceildiv(M,block_M),T.ceildiv(N,block_N); sub=block_M//2
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),RowBias:T.Tensor((M,),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N
            x=T.alloc_shared((sub,block_N),dtype); bc=T.alloc_shared((sub,block_N),dtype); one=T.alloc_shared((sub,),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.tile.tanh(x,x); T.tile.mul(x,x,2.0)
            T.copy(RowBias[rs:rs+sub],one); T.tile.broadcast(bc,one); T.tile.add(x,x,bc)
            T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main
if __name__=="__main__":
    torch.manual_seed(0); x=torch.randn(64,2048).npu(); b=torch.randn(64).npu(); out=level2_082_tanh_scale_bias(64,2048)(x,b); torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),(torch.tanh(x)*2+b[:,None]).cpu(),rtol=1e-2,atol=1e-2); print("level2_082 passed")
