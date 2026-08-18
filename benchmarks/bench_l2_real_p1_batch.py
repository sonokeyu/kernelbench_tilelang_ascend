"""Controlled timing of validated P1 static real epilogue kernels."""
import argparse, csv, importlib.util, os, time
import torch
import torch.nn.functional as F

ROOT="/workspace/tilelang-ascend/examples/elementwise/example_level2_real_p1_batch.py"
def event_ms(fn,args,warmup,iters):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize(); s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True); s.record()
    for _ in range(iters): fn(*args)
    e.record(); torch.npu.synchronize(); return s.elapsed_time(e)/iters

def main():
    p=argparse.ArgumentParser(); p.add_argument('--M',type=int,default=4096); p.add_argument('--N',type=int,default=8192); p.add_argument('--warmup',type=int,default=20); p.add_argument('--iters',type=int,default=100); p.add_argument('--out',required=True); a=p.parse_args()
    torch.manual_seed(0); x=torch.randn(a.M,a.N,dtype=torch.float32).npu(); rb=torch.randn(a.M,dtype=torch.float32).npu(); cm=torch.randn(a.N,dtype=torch.float32).npu()
    sp=importlib.util.spec_from_file_location('p1',ROOT); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); rows=[]
    cases=[('16','ConvTranspose2d_Mish_Add_Hardtanh_Scaling',mod.level2_016_add_hardtanh_scale,lambda:torch.clamp(x+.5,-1,1)*2,(x,)),('74','ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU',mod.level2_074_leaky_mul_leaky,lambda:F.leaky_relu(F.leaky_relu(x,.2)*rb[:,None],.2),(x,rb)),('88','Gemm_GroupNorm_Swish_Multiply_Swish',mod.level2_088_swish_mul_swish,lambda:(x*torch.sigmoid(x)*cm[None,:])*torch.sigmoid(x*torch.sigmoid(x)*cm[None,:]),(x,cm)),('91','ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid',mod.level2_091_bias_scale_sigmoid,lambda:torch.sigmoid((x+rb[:,None])*2),(x,rb))]
    for kid,op,factory,ref,call in cases:
        expected=ref(); torch_ms=event_ms(ref,(),a.warmup,a.iters); t0=time.perf_counter(); fn=factory(a.M,a.N); torch.npu.synchronize(); compile_ms=(time.perf_counter()-t0)*1000; actual=fn(*call); torch.npu.synchronize(); torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=1e-2,atol=1e-2); tile_ms=event_ms(fn,call,a.warmup,a.iters)
        row={'id':int(kid),'operator':op,'shape':f'{a.M},{a.N}','block_M':16,'block_N':1024,'warmup':a.warmup,'iters':a.iters,'torch_mean_ms':torch_ms,'tilelang_mean_ms':tile_ms,'compile_ms':compile_ms,'speedup_mean_torch_over_tilelang':torch_ms/tile_ms,'tilelang_passed':True,'variant':'arbitrary_input_real_fused_epilogue'}; rows.append(row); print(row,flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    with open(a.out,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
