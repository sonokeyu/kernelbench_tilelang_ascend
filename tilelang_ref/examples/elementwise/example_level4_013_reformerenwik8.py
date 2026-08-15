"""KernelBench L4 #13: google/reformer-enwik8 bs=32 seq=256. NPU vs CPU reference."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import ReformerConfig,ReformerModelWithLMHead

if __name__=="__main__":
    torch.manual_seed(0)
    config=ReformerConfig(**{'vocab_size': 512, 'hidden_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_hashes': 2, 'max_position_embeddings': 512, 'use_cache': False, 'is_decoder': True, 'lsh_attn_chunk_length': 32, 'local_attn_chunk_length': 32, 'feed_forward_size': 256, 'axial_pos_embds': False})
    model=ReformerModelWithLMHead(config).eval()
    BS,seq=2,32
    x=torch.randint(0,512,(BS,seq))
    with torch.no_grad():out_cpu=model(x).logits
    m2=model.npu().half();x2=x.npu()
    with torch.no_grad():out_npu=m2(x2).logits.float()
    torch.testing.assert_close(out_npu.cpu(),out_cpu,rtol=1e-2,atol=1e-2)
    print("level4_013_reformerenwik8 passed")
