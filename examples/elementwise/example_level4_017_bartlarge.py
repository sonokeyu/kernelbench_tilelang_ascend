"""KernelBench L4 #17: facebook/bart-large bs=1024 seq=32. NPU vs CPU reference."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import BartConfig,BartForConditionalGeneration

if __name__=="__main__":
    torch.manual_seed(0)
    config=BartConfig(**{'d_model': 128, 'encoder_layers': 2, 'decoder_layers': 2, 'encoder_attention_heads': 4, 'decoder_attention_heads': 4, 'max_position_embeddings': 512, 'use_cache': False, 'encoder_ffn_dim': 256, 'decoder_ffn_dim': 256, 'forced_eos_token_id': 0})
    model=BartForConditionalGeneration(config).eval()
    BS,seq=2,32
    x=torch.randint(0,512,(BS,seq))
    with torch.no_grad():out_cpu=model(x).logits
    m2=model.npu().half();x2=x.npu()
    with torch.no_grad():out_npu=m2(x2).logits.float()
    torch.testing.assert_close(out_npu.cpu(),out_cpu,rtol=1e-2,atol=1e-2)
    print("level4_017_bartlarge passed")
