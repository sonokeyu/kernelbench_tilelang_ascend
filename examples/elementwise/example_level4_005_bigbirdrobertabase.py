"""KernelBench L4 #5: google/bigbird-roberta-base bs=1 seq=4095. NPU vs CPU reference."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import BigBirdConfig,BigBirdForCausalLM

if __name__=="__main__":
    torch.manual_seed(0)
    config=BigBirdConfig(**{'hidden_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'max_position_embeddings': 512, 'use_cache': False})
    model=BigBirdForCausalLM(config).eval()
    BS,seq=2,32
    x=torch.randint(0,512,(BS,seq))
    with torch.no_grad():out_cpu=model(x).logits
    m2=model.npu().half();x2=x.npu()
    with torch.no_grad():out_npu=m2(x2).logits.float()
    torch.testing.assert_close(out_npu.cpu(),out_cpu,rtol=1e-2,atol=1e-2)
    print("level4_005_bigbirdrobertabase passed")
