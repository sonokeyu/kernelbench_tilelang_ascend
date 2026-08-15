"""KernelBench L4 #3: EleutherAI/gpt-neo-2.7B bs=1 seq=2047. NPU vs CPU reference."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPTNeoConfig,GPTNeoForCausalLM

if __name__=="__main__":
    torch.manual_seed(0)
    config=GPTNeoConfig(**{'hidden_size': 256, 'num_layers': 2, 'num_heads': 4, 'max_position_embeddings': 512, 'attention_types': [[['global', 'local'], 1]], 'use_cache': False})
    model=GPTNeoForCausalLM(config).eval()
    BS,seq=2,32
    x=torch.randint(0,512,(BS,seq))
    with torch.no_grad():out_cpu=model(x).logits
    m2=model.npu().half();x2=x.npu()
    with torch.no_grad():out_npu=m2(x2).logits.float()
    torch.testing.assert_close(out_npu.cpu(),out_cpu,rtol=1e-2,atol=1e-2)
    print("level4_003_gptneo27B passed")
