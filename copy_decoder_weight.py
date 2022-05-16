import torch
import os

ckpt_root_dir = '/tmp2/b07902050/multidoc2dial-mtl/ckpt'
model_path = os.path.join(ckpt_root_dir, 'rag-dpr-all-structure-mtl/pytorch_model.bin')
save_path = os.path.join(ckpt_root_dir, 'rag-dpr-all-structure/pytorch_model.bin')
decoder_prefix = 'rag.generator.model.decoder'
grounding_decoder_prefix = 'rag.generator.model.grounding_decoder'
generation_decoder_prefix = 'rag.generator.model.generation_decoder'
lm_head = 'rag.generator.lm_head.weight'
grounding_lm_head = 'rag.generator.grounding_lm_head.weight'
generation_lm_head = 'rag.generator.generation_lm_head.weight'

state_dict = torch.load(model_path, map_location="cpu")
state_dict[grounding_lm_head] = state_dict[lm_head]
state_dict[generation_lm_head] = state_dict[lm_head]
del state_dict[lm_head]

keys = state_dict.keys()
decoder_keys = [key for key in keys if decoder_prefix in 'rag.generator.model.decoder']

for key in decoder_keys:
    name = key.replace(decoder_prefix + '.', '')
    grounding_decoder_key = grounding_decoder_prefix + '.' + name
    generation_decoder_key = generation_decoder_prefix + '.' + name  
    state_dict[grounding_decoder_key] = state_dict[key].clone()
    state_dict[generation_decoder_key] = state_dict[key].clone()
    del state_dict[key]

torch.save(state_dict, save_path)