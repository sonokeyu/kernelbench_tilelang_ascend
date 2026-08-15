"""KernelBench L4 #19: gpt2 bs=1024 seq=32. NPU vs CPU reference."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2Config,GPT2LMHeadModel

if __name__=="__main__":
    torch.manual_seed(0)
    config=GPT2Config(**{'n_embd': 128, 'n_layer': 2, 'n_head': 4, 'n_positions': 512, 'use_cache': False})
    model=GPT2LMHeadModel(config).eval()
    BS,seq=2,32
    x=torch.randint(0,512,(BS,seq))
    with torch.no_grad():out_cpu=model(x).logits
    m2=model.npu().half();x2=x.npu()
    with torch.no_grad():out_npu=m2(x2).logits.float()
    torch.testing.assert_close(out_npu.cpu(),out_cpu,rtol=1e-2,atol=1e-2)
    print("level4_019_gpt2 passed")
