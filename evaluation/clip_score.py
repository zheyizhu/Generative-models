
import os
import torch
import numpy as np
import re
from torchvision.io import read_image

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import Dataset


classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image


@torch.no_grad()
def compute_clip_score(dataset: DataLoader, clip_model="ViT-B/32", device="cuda", how_many=5000):
    print("Computing CLIP score")
    import clip as openai_clip 
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load("ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    cos_sims = []
    count = 0
    for imgs, txts in tqdm(dataset):
        imgs_pil = [clip_preprocessor(tensor2pil(img)) for img in imgs]
        imgs = torch.stack(imgs_pil, dim=0).to(device)
        texts = list()
        for item in txts:
            texts.append(classes[item])
        tokens = openai_clip.tokenize(texts).to(device)
        # Prepending text prompts with "A photo depicts "
        # https://arxiv.org/abs/2104.08718
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[:, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(tokens.shape[0], -1)
        
        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat([new_text_tokens[:, :76], last_cols], dim=1)
        
        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = F.cosine_similarity(img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]
        if count >= how_many:
            break
    
    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    return clip_score


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = list()
        self.images = list()
        for filename in os.listdir(self.img_dir):
          f = os.path.join(self.img_dir, filename)
          if os.path.isfile(f):
              label = re.findall(r'\d+', filename)
              self.labels.append(torch.tensor(int(label[0]), dtype=torch.int8))
              
              image = read_image(f)
              if self.transform:
                  self.images.append(self.transform(image))
              else:
                  self.images.append(image)
                  
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

if __name__ == "__main__":
  transform_real = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  testset_real = torchvision.datasets.CIFAR10(root='../stablediffusion/data', train=False, download=True, transform=transform_real)
  testloader_real = torch.utils.data.DataLoader(testset_real, batch_size=128, shuffle=False, num_workers=2)

  img_dir = "../stablediffusion/fake_images/"
  transform_fake = transforms.Compose([
      # transforms.ToTensor(),
      transforms.ConvertImageDtype(torch.float32),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      # transforms.Resize(32)
  ])

  testset_fake = CustomImageDataset(img_dir = img_dir,transform = transform_fake)
  testloader_fake = torch.utils.data.DataLoader(testset_fake, batch_size=128, shuffle=True, num_workers=2)
  
  clip_score_real = compute_clip_score(testloader_real, how_many = 5000)
  clip_score_fake = compute_clip_score(testloader_fake, how_many = 1000)
  print("clip score real:", clip_score_real)
  print("clip score fake:", clip_score_fake)