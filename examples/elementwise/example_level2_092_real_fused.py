"""L2 #92 materialized GroupNorm output -> tanh -> simplified HardSwish -> residual."""
import tilelang
import tilelang.language as T
import torch
PC={tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE:True,tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:True}
@tilelang.jit(out_idx=[2],pass_configs=PC)
def level2_092_tanh_hardswish_residual(M,N,block_M=16,block_N=1024,dtype="float"):
    mn,nn=T.ceildiv(M,block_M),T.ceildiv(N,block_N); sub=block_M//2
    @T.prim_func
    def main(A:T.Tensor((M,N),dtype),Residual:T.Tensor((M,N),dtype),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(mn*nn,is_npu=True) as (cid,vid):
            bm,bn=cid//nn,cid%nn; rs,cs=bm*block_M+vid*sub,bn*block_N
            x=T.alloc_shared((sub,block_N),dtype); t=T.alloc_shared((sub,block_N),dtype); r=T.alloc_shared((sub,block_N),dtype)
            T.copy(A[rs:rs+sub,cs:cs+block_N],x); T.copy(Residual[rs:rs+sub,cs:cs+block_N],r); T.tile.tanh(x,x)
            T.tile.add(t,x,3.0); T.tile.mul(x,x,t); T.tile.div(x,x,6.0); T.tile.add(x,x,r); T.copy(x,Out[rs:rs+sub,cs:cs+block_N])
    return main
if __name__=="__main__":
    torch.manual_seed(0); x=torch.randn(64,2048).npu(); r=torch.randn(64,2048).npu(); out=level2_092_tanh_hardswish_residual(64,2048)(x,r); y=torch.tanh(x); ref=torch.nn.functional.hardswish(y)+r; torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),ref.cpu(),rtol=1e-2,atol=1e-2); print("level2_092 passed")
