import argparse
import csv
import importlib.util
import os
import time

import torch


def event_ms(fn,args,warmup,iters):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize(); s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True); s.record()
    for _ in range(iters): fn(*args)
    e.record(); torch.npu.synchronize(); return s.elapsed_time(e)/iters

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ids',default='50,30,96'); p.add_argument('--M',type=int,default=4096); p.add_argument('--N',type=int,default=8192); p.add_argument('--warmup',type=int,default=20); p.add_argument('--iters',type=int,default=100); p.add_argument('--out',required=True); a=p.parse_args()
    torch.manual_seed(0); x=torch.randn(a.M,a.N,dtype=torch.float32).npu(); b=torch.randn(a.M,dtype=torch.float32).npu()
    sp=importlib.util.spec_from_file_location('p5','/workspace/tilelang-ascend/examples/elementwise/example_level2_real_p5_batch.py'); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    cases={'50':(mod.level2_050_scale_bias_scale,lambda:x*.5+b[:,None],(x,b),'ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling'),'30':(mod.level2_030_post_gn_hardtanh,lambda:torch.clamp(x,-2,2),(x,),'Gemm_GroupNorm_Hardtanh'),'96':(mod.level2_096_post_gap_clamp,lambda:torch.clamp(x,0,1),(x,),'ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp')}
    rows=[]
    for kid in a.ids.split(','):
        factory,ref,call,op=cases[kid]; expected=ref(); torch_ms=event_ms(ref,(),a.warmup,a.iters); t0=time.perf_counter(); fn=factory(a.M,a.N); torch.npu.synchronize(); compile_ms=(time.perf_counter()-t0)*1000; actual=fn(*call); torch.npu.synchronize(); torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=1e-2,atol=1e-2); tile_ms=event_ms(fn,call,a.warmup,a.iters); row={'id':int(kid),'operator':op,'shape':f'{a.M},{a.N}','block_M':16,'block_N':1024,'warmup':a.warmup,'iters':a.iters,'torch_mean_ms':torch_ms,'tilelang_mean_ms':tile_ms,'compile_ms':compile_ms,'speedup_mean_torch_over_tilelang':torch_ms/tile_ms,'tilelang_passed':True,'variant':'arbitrary_input_real_fused_epilogue'}; rows.append(row); print(row,flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    with open(a.out,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
