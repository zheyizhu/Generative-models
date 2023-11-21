
from flickr_ae import AutoencoderKL


import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

import numpy

from utils import ImageLogger
import torch 
import data_pre
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModelWithProjection
ImageProcessorConfig = {
  "crop_size": {
    "height": 256,
    "width": 256
  },
  "do_center_crop": True,
  "do_convert_rgb": True,
  "do_normalize": True,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 256
  }
}
opt_kwargs = {}
transform = CLIPImageProcessor(**ImageProcessorConfig)
print(transform)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

args = {}
args['batch_size'] = 2
args['num_workers'] = 2
data_loader_train = data_pre.get_train_dataset(transform, tokenizer, args['batch_size'], args['num_workers'], opt_kwargs)
data_loader_test = data_pre.get_test_dataset(transform, tokenizer, args['batch_size'], args['num_workers'], opt_kwargs)
data_loader_val = data_pre.get_val_dataset(transform, tokenizer, args['batch_size'], args['num_workers'], opt_kwargs)
# "pixel_values", "input_ids", 'image_id', 'attention_mask'

cond_stage_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")#GPT2Model.from_pretrained("gpt2")
cond_stage_model.to('cuda')

# 16*16*16
# ddconfig = { "double_z": True, "resolution": 256, "in_channels": 3,"out_ch": 3, "ch": 128, "z_channels": 16, "ch_mult": [ 1,1,2,2,4 ] ,"num_res_blocks": 2,"attn_resolutions": [16],"dropout": 0.0}
# 64*8*8
# ddconfig = { "double_z": True, "resolution": 256, "in_channels": 3,"out_ch": 3, "ch": 128, "z_channels": 64, "ch_mult": [ 1,1,2,2,4,4 ] ,"num_res_blocks": 2,"attn_resolutions": [16,8],"dropout": 0.0}
# 3x64x64
ddconfig = { "double_z": True, "resolution": 256, "in_channels": 3,"out_ch": 3, "ch": 128, "z_channels": 3, "ch_mult": [ 1,2,4 ] ,"num_res_blocks": 2,"attn_resolutions": [],"dropout": 0.0}
lossconfig = {"disc_start": 50001, "kl_weight": 0.000001, "disc_weight": 0.5}
AutoEncoder = AutoencoderKL(ddconfig, lossconfig = lossconfig, embed_dim=3, monitor = "val/rec_loss", )
AutoEncoder.init_from_ckpt("model_36464.ckpt")
AutoEncoder.to('cuda')


# xc = tokenizer([""]*1, padding=True, return_tensors="pt")
# for key in xc:
#     xc[key] = torch.Tensor(xc[key]).long().to('cuda')
# print(cond_stage_model(**xc).text_embeds)

text = []
images = []
original_images = []
acc = 0
for batch_idx, samples in enumerate(data_loader_val):
    input = {}
    input['input_ids'] = torch.Tensor(samples['input_ids']).long().to(cond_stage_model.device)
    input['attention_mask'] = torch.Tensor(samples['attention_mask']).long().to(cond_stage_model.device)
    pooled_output = cond_stage_model(**input).text_embeds
    text.append(pooled_output.detach().cpu())

    x = torch.tensor(numpy.array(samples['pixel_values'])).to(AutoEncoder.device)
    x = torch.reshape(x, (x.shape[0],x.shape[2],x.shape[3],x.shape[4]))
    original_images.append(x.detach().cpu())
    z = AutoEncoder.encode(x).sample()
    images.append(z.detach().cpu())
    acc+=1
    print(acc)
    # if acc == 50:
    #   break

text_embs = torch.cat(text)
vision_embs = torch.cat(images)
original_images = torch.cat(original_images)
print(text_embs.shape)
print(vision_embs.shape)
print(original_images.shape)


torch.save(text_embs, './flickr_data/36464_flickr_text_embs_clip_val.pt')
torch.save(vision_embs, './flickr_data/36464_flickr_vision_embs_clip_val.pt')
torch.save(original_images, './flickr_data/36464_flickr_original_images_clip_val.pt')


