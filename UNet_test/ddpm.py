
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from flickr_ae import AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch_fidelity



from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


from main import instantiate_from_config

from contextlib import contextmanager, nullcontext


class RealFlickrDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        self.images = torch.load(data_path)
        self.images = (self.images * 255).type(torch.uint8)

    def __getitem__(self, index: int):
        data = self.images[index]
        return data

    def __len__(self) -> int:
        return self.images.shape[0]

class FakeFlickrDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.images = samples
        # self.images = (self.images * 255).type(torch.uint8)

    def __getitem__(self, index: int):
        data = self.images[index]
        return data

    def __len__(self) -> int:
        return self.images.shape[0]
    

uc_clip = [ 2.3279e-01, -7.1315e-02,  4.2952e-01, -5.4408e-01,  5.5927e-02,
          6.6773e-02,  1.2999e-01, -2.6866e-01,  3.4797e-01, -3.9918e-01,
         -4.6921e-02, -4.1369e-02, -5.0848e-01,  2.4111e-01, -2.3073e-01,
          7.9518e-02,  3.4198e-01,  1.9837e-01, -1.8987e-01,  4.2727e-02,
          2.6027e-01, -1.3141e-01, -5.8932e-02, -8.7270e-02,  2.2419e-01,
         -5.3481e-01,  3.8202e-01,  7.4324e-02,  7.9764e-01,  2.3768e-01,
         -2.7833e-01, -1.5711e-01, -4.1629e-01, -2.3820e-01, -3.5428e-01,
          3.7521e-01,  1.0446e-01, -5.2568e-01, -1.6852e-01, -5.3294e-01,
         -1.0642e-01, -1.1299e-01,  1.6070e-01,  1.2259e-01,  2.4066e-01,
         -4.5185e-01,  8.1162e-02, -1.0159e-01, -2.9277e-01, -4.8451e-01,
         -6.8222e-02, -1.6054e-01,  1.5700e-01, -7.3122e-02,  8.2711e-02,
         -1.5224e-01, -3.2179e-02,  6.5815e-02,  2.5309e-01, -2.6684e-02,
         -6.6725e-02,  2.1458e-01, -1.1665e-01,  2.2089e-01, -8.9376e-02,
          2.3203e-01,  5.1356e-01, -6.7778e-02, -2.7126e-02, -4.6637e-01,
         -1.6710e-01,  9.1149e-01, -3.9966e-02,  1.4519e-01, -1.7772e-01,
          4.0394e-01, -7.0657e-02,  1.9163e-01, -1.7777e-02, -6.1727e-01,
         -2.6419e-01,  2.2247e-02,  1.6718e-01, -1.3280e-01,  5.1039e-03,
          4.0271e-01, -8.2014e-02,  1.5915e-01, -4.3231e-01,  8.8839e-01,
         -6.8189e-02,  3.8690e-01, -3.1712e-01,  3.2383e-01, -3.0963e-01,
          2.6199e-02,  2.8252e-01,  3.2557e-02,  4.9611e-01,  3.1193e-02,
         -3.3832e-01, -2.4828e-01, -2.4750e-01, -4.2598e-02,  1.4659e-01,
          1.0973e-01,  6.5427e-02, -4.0166e-01,  2.5461e-01, -3.0163e-01,
         -9.8969e-02, -5.4393e-02, -2.3696e-01,  1.5346e-01, -9.9872e-02,
          1.0542e-01, -7.0868e-01, -3.7217e-01, -1.9338e-02, -3.1761e-02,
          6.4891e-02, -4.4186e-01, -4.1661e-04,  1.3601e-01, -7.1724e-02,
         -9.1193e-02,  3.8522e-02,  2.0355e-01, -3.0403e-01, -3.7738e-01,
          2.6820e-01, -8.4630e-01,  3.6883e-02, -4.5413e-01,  2.9464e-01,
          5.0426e-01,  2.2601e-01,  5.7431e-01, -2.1424e-01,  2.8149e-01,
          7.0491e-01,  6.5848e-02,  6.8959e-02, -3.4007e-01,  7.4364e-02,
          1.1425e-01,  5.7236e-01, -4.7381e-01, -8.4422e-01,  4.1260e-02,
          1.2253e-01, -2.7325e-01,  6.3257e-01, -1.5002e-01,  5.3715e-01,
         -5.7549e-02,  3.4721e-01,  2.5902e-01,  4.1923e-02,  1.7065e-01,
          2.7834e-01,  4.1549e-01,  2.6877e-02,  3.5346e-01,  1.9371e-02,
         -1.7761e-01,  3.9265e-01, -2.0409e-01,  3.4766e-01, -6.0700e-01,
          2.1137e-01, -5.3207e-01,  3.9336e-01, -9.1554e-02, -1.5208e-02,
         -2.0491e-01,  1.5087e-01, -1.8344e+00,  3.1286e-01,  2.1304e-01,
          1.0792e+00,  8.8291e-02, -6.2158e-01, -3.4201e-02, -1.7017e-01,
         -9.1834e-02, -1.0215e-01,  1.2016e-01, -1.4309e-02, -3.9443e-01,
         -3.7891e-01, -2.3694e-01,  2.1819e-01, -1.6213e-02,  1.4842e-01,
         -2.8816e+00, -5.9387e-02,  5.1363e-01, -4.4727e-02,  8.8851e-02,
          2.8659e-01,  3.4143e-02, -1.1068e-01,  1.1802e-01,  1.0728e-01,
         -2.1365e-02, -2.2338e-01, -5.8715e-02,  6.7943e-01, -3.5256e-02,
         -2.9322e-02, -4.5095e-02, -1.3633e-01,  3.4152e-01,  2.0124e-01,
          2.6659e-01,  2.9110e-01,  3.0518e-01,  8.8727e-02,  6.1221e-01,
          1.4568e-01, -4.0983e-01,  8.6634e-03, -7.8147e-02,  6.4514e-02,
         -2.2122e-02, -1.5405e-01,  7.9482e-02,  5.4675e-01, -1.1840e-01,
          2.0714e-01,  2.1866e-02, -2.0695e-01, -4.1785e-01,  1.5678e-01,
         -3.0092e-01, -3.9874e-01,  1.9776e-01,  5.8354e-01, -1.6871e-01,
          3.9941e-01, -7.0260e-02,  2.7885e-01,  1.6402e-01,  6.1138e-03,
          8.7020e-01,  2.9377e-02,  3.7463e-01,  5.5311e-02,  6.3770e-01,
          1.2910e-01,  2.2605e-04,  2.2765e-01, -3.0432e-01,  2.8193e-01,
          2.7598e-01, -7.4516e-02,  5.1160e-01,  2.2811e-01,  1.0716e+00,
         -2.5904e-01,  7.7325e-02, -1.1727e-02,  2.5190e-01,  2.6683e-01,
          2.0120e-01,  1.7729e-02, -7.3227e-01,  2.8284e-02, -2.4866e-01,
         -4.6139e-01,  1.3951e-02,  3.0472e-01, -3.3239e-01, -2.2646e-01,
         -4.3243e-01, -9.2285e-03,  1.1210e-01,  2.4956e-01,  1.0286e-01,
         -3.2063e-01, -1.0157e+00, -3.1144e-01,  1.0534e-01,  1.5500e-01,
          1.4063e-01,  4.1821e-01,  1.4103e-01, -2.7669e+00, -5.6876e-02,
         -2.0032e-01, -8.4563e-02, -9.2855e-01,  8.6689e-02, -4.0132e-03,
         -1.0748e-01, -5.6338e-01,  2.9139e-01,  2.8842e-01, -2.3533e-01,
         -2.7075e-01, -1.5105e-01, -3.4584e-01, -1.6884e-01,  1.3275e-01,
         -3.3493e-02, -9.3893e-02, -9.3794e-02,  2.8272e-01, -1.5730e-01,
         -6.6981e-01, -1.3770e-01, -8.0900e-02, -2.0049e-01,  4.2372e-01,
         -2.1216e-01, -1.4059e+01,  1.6747e-01, -6.2217e-01,  1.0549e-01,
         -5.5593e-02, -7.8499e-01,  2.8457e-01,  1.9277e-01, -3.2058e-01,
          4.4604e-01, -1.0070e+00,  8.7568e-03,  6.1493e-02, -1.9848e-01,
         -1.4303e-02,  2.4688e-01, -1.4490e-01, -2.3571e-01,  4.1761e-01,
          1.1520e-01, -1.5425e-01,  4.4007e-02, -4.6418e-01, -6.1833e-01,
         -1.0436e-01, -2.0370e-01, -8.2259e-02,  1.7054e-01,  1.4260e-02,
         -1.2559e-01,  2.3875e-01,  8.8418e-02,  3.8065e-01,  2.6411e-01,
         -2.1290e-02, -1.2580e-01, -6.7955e-02,  5.0091e-01,  4.4652e-01,
         -2.1524e-01,  8.6798e-01, -9.1576e-02, -3.0095e-02, -3.3847e-01,
          8.0581e-02,  1.8507e-01,  8.3891e-01,  3.8478e-01,  3.8252e-01,
          3.9593e-01,  4.6459e-01,  2.9624e-01, -3.9938e-02,  3.4222e-01,
          1.0304e-01,  2.8255e-01,  1.0469e-01, -6.3757e-02, -1.1819e-01,
          7.5523e-02, -4.0828e-01, -2.5528e-02,  2.2315e-01, -1.9851e-01,
          3.8945e-01, -4.0615e-01, -2.2698e-01,  7.2242e-02,  2.3525e-01,
         -3.8438e-01,  3.9513e-01, -1.4025e-01,  3.9621e-02,  1.8607e-01,
          3.0175e-01, -1.2398e-01, -2.9090e-01,  1.6614e-02, -4.6875e-01,
         -5.5083e-01, -9.0355e-02,  1.1316e-01, -3.3593e-02,  4.3141e-01,
          6.1043e-01,  2.5846e-01,  8.0126e-02, -1.6359e-01,  1.4396e+00,
          2.2315e-01,  1.6899e-01, -3.7965e-02,  5.9673e-02,  4.0204e-01,
          1.2187e-01, -5.5850e-01,  1.4602e-01, -1.5020e+00, -5.7354e-03,
          2.1861e-01, -3.1137e-01, -1.8800e-01, -4.1111e-01, -1.4711e-01,
         -4.3922e-01,  5.1525e-02,  1.3297e-01, -2.1232e-01, -1.6509e-01,
          1.5294e-01, -9.3775e-02, -2.7330e-02,  3.1403e-01,  2.2433e-01,
         -2.3561e-01,  1.9918e-01,  2.9848e-01, -9.3354e-02,  3.8691e-01,
          9.6541e-01, -2.0393e-01, -9.5444e-02, -1.6617e-01,  2.5620e-01,
          1.2852e+01,  1.9602e-04,  1.4626e-01,  4.0329e-01, -2.8448e-01,
         -7.2512e-02,  5.6196e-01,  2.8935e-01,  4.1035e-02, -3.5804e-01,
         -1.4409e+00, -3.4863e-01,  7.8011e-01,  1.1457e-01, -1.9564e-01,
          7.9264e-01,  4.2934e-01,  3.9345e-02, -3.3445e-01,  2.8785e-01,
         -3.3175e-01,  1.5686e-01,  1.4777e-01,  5.2868e-01,  6.7249e-03,
         -1.6528e-01,  2.2426e-01,  5.6676e-01,  2.7320e-01,  2.1062e-01,
          1.6412e-01, -1.1292e-01,  2.7777e-01, -2.4146e-02,  4.8704e-02,
         -3.1579e-01,  3.6245e-01,  6.6899e-02, -4.9211e-01, -1.6412e-01,
         -1.7496e-01, -3.6210e-01, -4.1687e-01, -5.1291e-01, -3.8262e-01,
          2.0400e-01,  3.1320e-01, -6.0975e-01,  1.4347e-01,  5.2146e-01,
          1.7234e-02,  2.5157e-01, -7.6172e-01, -1.9530e-01, -1.4541e-02,
          8.1289e-02,  4.9901e-02, -4.9549e-01,  7.4812e-02,  1.3068e-01,
          2.3324e-01, -4.7717e-01, -1.8984e-02,  2.2168e-01, -3.5567e-01,
         -1.5578e-01, -1.9763e-01, -3.0416e-01, -9.4404e-01,  1.3455e-02,
          2.4683e-01, -4.4160e-01, -5.1174e-02,  4.4747e-02, -9.0029e-03,
          4.5811e-02, -8.8027e-03, -3.9886e-01, -1.7533e-01, -3.8969e-01,
          2.2158e-01, -4.5404e-01, -2.3309e-01,  7.5750e-03,  5.1734e-02,
          5.5368e-01,  2.1680e-01,  2.6423e-01, -1.9306e+00, -3.9690e-01,
         -5.2426e-01, -4.9502e-01,  7.2482e-02, -7.2784e-02,  3.9504e-01,
          9.3172e-03, -2.5902e-01,  2.4524e-01, -9.1787e-01,  2.6630e-03,
         -1.6214e-02,  2.2911e-01, -7.8394e-03,  5.1218e-03, -2.9821e-01,
         -6.1370e-02,  1.8453e-01, -9.8656e-02,  1.3961e-01,  1.2435e-01,
          4.3008e-01, -2.6651e-01,  5.5610e-02, -1.4299e-02,  8.4917e-02,
          3.9034e-01,  3.1639e-02,  3.6006e-01,  6.5088e-02, -9.0882e-02,
          5.1749e-01,  5.3885e-01, -2.5108e-01,  1.4823e-01,  8.8497e-01,
          3.6254e-01,  1.9327e-01,  9.7607e-03, -3.1013e-02,  3.4243e-01,
          4.0658e-01, -5.6296e-01, -2.7520e-01, -2.2484e-01,  3.0418e-01,
         -4.8823e-01,  3.0851e-01,  1.1513e-01, -1.3862e-02, -2.9064e-01,
          2.9677e-01,  7.4959e-03, -2.3553e-01,  2.4834e-01, -4.9819e-01,
         -6.7230e-01,  2.2359e-01, -1.7551e-01, -1.0625e-01, -2.4911e-01,
          9.2651e-02,  1.7177e-01, -2.5625e-01, -1.1038e-01,  4.7263e-01,
          2.6437e-01,  1.6097e-01, -2.8210e-01,  9.9044e-02,  2.6170e-01,
         -2.5807e-01,  4.1917e-01,  5.0313e-02,  2.6889e-01,  3.2195e-03,
          5.1070e-01,  1.3332e-01,  1.9589e-01, -3.2298e-02, -2.7248e-02,
          5.5698e-02, -8.6636e-03, -2.7837e-01, -1.6460e-02, -1.8817e-01,
         -1.0558e-01,  1.8614e-01, -9.7648e-01, -5.2875e-02,  6.4193e-02,
          2.1716e-01,  3.0930e-02, -1.3324e-01,  3.0654e-01, -6.7733e-01,
          5.8321e-01,  2.6934e-01, -2.1715e-01,  3.0824e-01, -1.0414e-01,
         -1.2811e-01,  5.9077e-01, -5.6839e-02, -1.2055e-01,  9.1507e-02,
          1.8890e-01,  2.7215e-01, -6.7459e-01, -5.4084e-02, -1.3915e-01,
         -8.0074e-02,  2.2354e-01, -3.8407e-02, -2.8770e-01,  1.5078e+00,
          9.8587e-02,  1.6354e-01, -1.8530e-01,  6.3487e-01, -7.3507e-02,
          3.3969e-01, -3.9594e-01, -3.7405e-01,  8.1988e-02,  4.7816e-01,
          1.6448e-01,  1.3778e-01, -2.2859e-01, -6.9853e-02, -8.3608e-01,
         -5.3637e-01, -2.9796e-01, -2.2860e-01,  2.7582e-01, -4.5530e-02,
          7.9136e-03,  2.2744e-01, -1.4535e-01, -1.6535e-01,  1.8512e-01,
         -2.6589e-01,  3.1557e-01,  1.8533e-01,  1.8273e-01, -1.9571e-01,
         -2.7632e-01,  2.8814e-01, -1.9738e-01, -1.0670e-01, -2.9315e-01,
         -2.5179e-01,  4.1770e-02, -3.7973e-01, -3.9476e-01,  4.7382e-01,
          5.1261e-01, -3.2481e-01,  1.7176e-01,  1.5590e-01,  1.3140e-02,
          3.9619e-01,  1.2163e-01, -2.6879e-01, -1.0146e-01,  3.2987e-01,
          6.4182e-02, -7.5193e-02,  3.0818e-01, -8.3565e-03, -3.0668e-01,
          1.1668e-01,  9.3473e-02, -1.9000e-02, -1.0820e+00,  3.5092e-01,
          4.4161e-01, -1.6675e-01, -7.9382e-02, -4.7475e-03, -2.5889e-01,
         -3.7033e-01,  5.6853e-01, -1.0937e-01,  3.9099e-01,  4.6025e-01,
         -2.6163e-01, -2.5863e-02, -1.7309e-01,  8.9179e-02,  1.3975e-01,
         -3.6571e-02,  6.0632e-03,  3.1803e-01, -6.6787e-01,  9.2360e-01,
          6.3490e-01,  1.3421e-01,  7.5724e-02, -5.6421e-01,  2.2731e-01,
         -4.7060e-02,  2.0744e-01,  2.2903e-01, -2.2104e-01,  6.5041e-01,
          1.8345e-01,  4.4748e-01,  1.9019e-01,  3.3817e-01,  4.9230e-01,
          2.7776e-01,  6.5564e-01,  2.8638e-03,  4.9954e-01,  7.1009e-02,
         -1.3422e-01,  2.6388e-02, -5.5161e-02,  1.2072e-02, -4.9968e-01,
         -2.2555e-01, -2.1714e-01,  2.6704e-01,  7.2181e-02, -3.4131e-01,
         -4.1473e-01,  4.9306e-01,  3.3943e-01, -2.1635e-01, -3.8134e-01,
         -1.3211e-01, -6.8421e-01,  6.9213e-01,  3.0949e-01,  7.6198e-01,
         -2.5276e-01, -4.9041e-01,  1.0734e-01]

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



class TestUnetDDPM(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 first_stage_config,
                #  cond_stage_model,
                 in_channels,
                 input_scale_factor=1.0,
                 timesteps=1000,
                 image_size=32,
                 log_every_t=200,
                 linear_start=0.00085,
                 linear_end=0.0120,
                 ddim_step=None,
                 learning_rate=3.2e-5,
                 monitor="val/loss",
                 val_image_datapath="",
                 val_cond_path=""
                 ):
        super().__init__()
        if monitor is not None:
            self.monitor = monitor
        self.learning_rate = learning_rate
        self.log_every_t = log_every_t
        self.image_size = image_size  
        self.channels = in_channels
        self.num_timesteps = int(timesteps)
        self.input_scale_factor = input_scale_factor

        self.model = instantiate_from_config(unet_config)

        # self.cond_stage_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")
        # self.cond_stage_model = self.cond_stage_model.eval()
        # self.cond_stage_model.train = disabled_train
        # for name, param in self.cond_stage_model.named_parameters():
        #     param.requires_grad = False
        self.first_stage_model = AutoencoderKL(**first_stage_config)
        self.first_stage_model.init_from_ckpt(first_stage_config['ckpt_path'])
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for name, param in self.first_stage_model.named_parameters():
            param.requires_grad = False

        self.ddim_step = ddim_step
    
        self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        self.make_schedule_ddim(self.ddim_step)
        
        # for validation:
        # self.val_cond = list()
        # self.val_image = list()
        self.val_image_datapath = val_image_datapath
        self.val_cond_path = val_cond_path
        self.clip_score_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_score_image_encoder = self.clip_score_image_encoder.eval()
        self.clip_score_image_encoder.train = disabled_train
        for name, param in self.clip_score_image_encoder.named_parameters():
            param.requires_grad = False
        self.clip_score_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


    def register_schedule(self, timesteps=1000,linear_start=1e-4, linear_end=2e-2):
        #linear schedule
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2).numpy()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

        # predict x_t directly from noise
        self.register_buffer('sqrt_1_over_alphas', to_torch(np.sqrt(1. / alphas))) #
        self.register_buffer('sqrt_betas', to_torch(np.sqrt(betas))) #
        self.register_buffer('one_minus_alpha_over_sqrt_one_minus_alpha_cumprod', to_torch((1.0-alphas)/np.sqrt(1. - alphas_cumprod))) 
        
        # predict x0 from noise
        self.register_buffer('sqrt_1_over_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod))) #
        self.register_buffer('sqrt_1_over_alphas_cumprod_minus_1', to_torch(np.sqrt(1. / alphas_cumprod - 1.))) #


        # q(x_{t-1} | x_t, x_0)
        posterior_variance =  betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  #
        self.register_buffer('posterior_variance', to_torch(posterior_variance)) #
        self.register_buffer('log_posterior_variance', to_torch(np.log(posterior_variance))) #
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))) #
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))) #

        

    


############################################## training #############################################


    # @torch.no_grad()
    # def get_learned_conditioning(self, c):
    #     last_hidden_state = self.cond_stage_model(**c).last_hidden_state
    #     pooled_output = last_hidden_state[
    #             torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
    #             c['input_ids'].to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    #         ]
    #     return pooled_output

    
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        pooled_output = torch.Tensor(uc_clip).to(self.device)
        pooled_output = pooled_output.repeat(batch_size, 1)
        return pooled_output
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        z = torch.reshape(z,(z.shape[0],self.channels,self.image_size,self.image_size))
        z = 1. / self.input_scale_factor * z
        x_rec = self.first_stage_model.decode(z)
        # x_rec = z
        return x_rec

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, bs=None):
        z = batch['vision_embedding'].to(self.device)* self.input_scale_factor
        c = batch['text_embedding'].to(self.device)
        # x = batch['original_image'].to(self.device)
   
        if bs is not None:
            c = c[:bs]
            z = z[:bs]
            
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([z, xrec])
        return out
    

    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x, cond):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        
        model_output = self.model(x_noisy, t, cond)
        target = noise
        
        loss = torch.nn.functional.mse_loss(model_output, target, reduction="none").mean()
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)
        z, c = self.get_input(batch)
        loss, loss_dict = self(z, c)
        self.log_dict(loss_dict, prog_bar=True,logger=True, on_step=True, on_epoch=True)
        return loss



############################################## validation #############################################

    def validation_step(self, batch, batch_idx):
        z, c, x, xrec= self.get_input(batch, return_first_stage_outputs=True)
        loss, loss_dict = self(z, c)        
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return 
        
    def validation_epoch_end(self, outputs):
        print("current_epoch:", self.current_epoch)
        if (self.current_epoch+1) % 10 != 0:
            return
        
        metric_dict = {}
        
        conds = torch.load(self.val_cond_path)
        conds = conds.to(self.device)
        
        fake_images = list()
        for cond in torch.split(conds, 32, dim=0):
            bs = cond.shape[0]
            uc = self.get_unconditional_conditioning(bs)
            samples_cfg_ddim, _ = self.sample_log(cond=cond, batch_size=bs,
                                                unconditional_guidance_scale=2.0,
                                                unconditional_conditioning=uc, use_ddim = True)
            x_samples_cfg_ddim = self.decode_first_stage(samples_cfg_ddim)
            fake_samples = torch.clamp((x_samples_cfg_ddim+1.0)/2.0, min=0.0, max=1.0)
            fake_images.append(fake_samples.detach().cpu())
        fake_samples = torch.cat(fake_images, 0)
        fake_samples = (fake_samples * 255).type(torch.uint8)
        fake_samples = fake_samples.detach()
        
        real_dataset = RealFlickrDataset(self.val_image_datapath)
        fake_dataset = FakeFlickrDataset(fake_samples.cpu().detach())
        
        metric_dict = torch_fidelity.calculate_metrics(
                                input1=fake_dataset, 
                                input2=real_dataset,
                                cuda=True, 
                                isc=True, 
                                fid=True, 
                                prc=True, 
                                verbose=True)        
        
        clip_score = self.compute_clip_score(fake_samples, conds)
        metric_dict.update({'clip_score': clip_score})
        
        del fake_samples
        del real_dataset
        del fake_dataset
        
        print(metric_dict)
        self.log_dict(metric_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        

    def compute_clip_score(self, images, text_emb):
        cos_sims = []
        count = 0
        imgs = torch.split(images, 32, dim=0)
        conds = torch.split(text_emb, 32, dim=0)
        for index, (img,cond) in enumerate(zip(imgs,conds)):
            inputs = self.clip_score_image_processor(img, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
            img_embs = self.clip_score_image_encoder(**inputs).image_embeds
            text_embs = cond
            similarities = torch.nn.functional.cosine_similarity(img_embs, text_embs, dim=1)
            cos_sims.append(similarities)
            count += similarities.shape[0]
            
        clip_score = torch.cat(cos_sims, dim=0).mean()
        clip_score = clip_score.detach().cpu()
        return clip_score



############################################## ddpm sampling #############################################

    @torch.no_grad()
    def p_sample(self, x_t, c, t, unconditional_guidance_scale=None,unconditional_conditioning=None):
        if unconditional_guidance_scale is not None and unconditional_conditioning is not None:
            c_uc = torch.cat([unconditional_conditioning, c])
            t_uc = torch.cat([t] * 2)
            x_t_uc = torch.cat([x_t] * 2)
            vision_output = self.model(x_t_uc, t_uc, c_uc)
            uncond_predicted_noise, cond_predicted_noise = vision_output.chunk(2)
            predicted_noise = uncond_predicted_noise + unconditional_guidance_scale * (cond_predicted_noise - uncond_predicted_noise)
        else:
            predicted_noise = self.model(x_t, t, c)
            
        predicted_x_0 = (extract_into_tensor(self.sqrt_1_over_alphas_cumprod, t, x_t.shape) * x_t -
               extract_into_tensor(self.sqrt_1_over_alphas_cumprod_minus_1, t, x_t.shape) * predicted_noise)

        noise = torch.randn(x_t.shape, device=x_t.device)
        # q(x_{t-1} | x_t, x_0)
        x_t_mean = (extract_into_tensor(self.posterior_mean_coef1, t, predicted_x_0.shape) * predicted_x_0 +
                    extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        log_x_t_var = extract_into_tensor(self.log_posterior_variance, t, x_t.shape)
        predicted_x_t = x_t_mean+torch.exp(0.5*log_x_t_var)*noise
        # predicted_x_t = extract_into_tensor(self.sqrt_1_over_alphas, t, x_t.shape) * (x_t - extract_into_tensor(self.one_minus_alpha_over_sqrt_one_minus_alpha_cumprod, t, x_t.shape) * predicted_noise )+ extract_into_tensor(self.sqrt_betas, t, x_t.shape) * noise
        return predicted_x_t, predicted_x_0


    @torch.no_grad()
    def sample_loop(self, cond, shape, return_intermediates=False,
                    unconditional_guidance_scale=None,unconditional_conditioning=None):
        # x_T -> x_{T-1} -> ... -> ... -> x_1 -> x_0
        log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        device = self.betas.device

        
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(1, timesteps)), desc='Sampling t', total=timesteps):
            ts = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img, x0 = self.p_sample(img, cond, ts, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if i % log_every_t == 1 or i == timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img







############################################## ddim sampling #############################################
    @torch.no_grad()
    def p_sample_ddim(self, x_t, c, t, unconditional_guidance_scale=None,unconditional_conditioning=None):
        if unconditional_guidance_scale is not None and unconditional_conditioning is not None:
            c_uc = torch.cat([unconditional_conditioning, c])
            t_uc = torch.cat([t] * 2)
            x_t_uc = torch.cat([x_t] * 2)
            vision_output = self.model(x_t_uc, t_uc, c_uc)
            uncond_predicted_noise, cond_predicted_noise = vision_output.chunk(2)
            predicted_noise = uncond_predicted_noise + unconditional_guidance_scale * (cond_predicted_noise - uncond_predicted_noise)
        else:
            predicted_noise = self.model(x_t, t, c)
        
        t = t//self.ddim_step
        predicted_x_0 = (x_t - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_ddim, t, x_t.shape) * predicted_noise)/extract_into_tensor(self.sqrt_alphas_cumprod_ddim, t, x_t.shape)
        self.eta = 0.
        sigma = self.eta * extract_into_tensor(self.sigmas, t, x_t.shape)
        direction_to_x_t = torch.sqrt(1.-extract_into_tensor(self.alphas_cumprod_prev_ddim, t, x_t.shape) - sigma**2) * predicted_noise
        noise = torch.randn(x_t.shape, device=x_t.device)
        random_noise = self.eta * sigma * noise
        predicted_x_t = extract_into_tensor(self.sqrt_alphas_cumprod_prev_ddim, t, x_t.shape) * predicted_x_0 + direction_to_x_t + random_noise
        return predicted_x_t, predicted_x_0
    
    def make_schedule_ddim(self, ddim_step = None):
        # ddim sampling
        if ddim_step == self.ddim_step:
            return
        if ddim_step is not None:
            self.ddim_step = ddim_step
        assert self.ddim_step is not None
        ddim_timesteps = np.asarray(list(range(self.ddim_step-1, self.num_timesteps+1, self.ddim_step)))
        # [49 99 149 199 249 299 349 399 449 499 .... 949 999]
        alphas_cumprod = self.alphas_cumprod.cpu()
        alphas_cumprod_ddim = alphas_cumprod[ddim_timesteps]
        alphas_cumprod_prev_ddim = np.asarray([alphas_cumprod[0]] + alphas_cumprod[ddim_timesteps[:-1]].tolist())
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('sqrt_one_minus_alphas_cumprod_ddim', to_torch(np.sqrt(1. - alphas_cumprod_ddim)).to(self.device))
        self.register_buffer('alphas_cumprod_prev_ddim' , to_torch((alphas_cumprod_prev_ddim)).to(self.device))
        self.register_buffer('sqrt_alphas_cumprod_ddim' , to_torch(np.sqrt(alphas_cumprod_ddim)).to(self.device))
        self.register_buffer('sqrt_alphas_cumprod_prev_ddim', to_torch(np.sqrt(alphas_cumprod_prev_ddim)).to(self.device))
        sigmas = np.sqrt((1 - alphas_cumprod_prev_ddim) / (1 - alphas_cumprod_ddim) * (1 - alphas_cumprod_ddim / alphas_cumprod_prev_ddim))
        self.register_buffer('sigmas', to_torch(sigmas).to(self.device))
        
    
    @torch.no_grad()
    def sample_loop_ddim(self, cond, shape, return_intermediates=False, ddim_step=None, 
                         unconditional_guidance_scale=None,unconditional_conditioning=None):
        # x_T -> x_{T-1} -> ... -> ... -> x_1 -> x_0
        self.make_schedule_ddim(ddim_step)
        log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        device = self.betas.device
        
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(self.ddim_step-1, self.num_timesteps+1, self.ddim_step)), desc='DDIM Sampling ', total=int(timesteps/self.ddim_step)):
            ts = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img, x0 = self.p_sample_ddim(img, cond, ts, 
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning)
            if (i+1) % log_every_t == 0 or i == self.ddim_step-1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, shape=None, 
               use_ddim = False, ddim_step=None,
               unconditional_guidance_scale=None,unconditional_conditioning=None,
               return_intermediates=False, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if use_ddim:
            return self.sample_loop_ddim(cond[:batch_size],shape,
                                        return_intermediates=return_intermediates, ddim_step=ddim_step,
                                        unconditional_guidance_scale=unconditional_guidance_scale, 
                                        unconditional_conditioning=unconditional_conditioning)
        else:
            return self.sample_loop(cond[:batch_size],shape,
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning,
                                    return_intermediates=return_intermediates)



    





############################################## image logging ##########################################

    @torch.no_grad()
    def sample_log(self, cond, batch_size, use_ddim = False, ddim_step=None, 
                   unconditional_guidance_scale=None,unconditional_conditioning=None, **kwargs):
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                            use_ddim = use_ddim, ddim_step=ddim_step,
                                            unconditional_guidance_scale=unconditional_guidance_scale, 
                                            unconditional_conditioning=unconditional_conditioning,
                                            return_intermediates=True, **kwargs)
        return samples, intermediates
    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid


    @torch.no_grad()
    def progressive_denoising(self, cond, shape, 
                              batch_size=None, x_T=None, start_T=None,
                              log_every_t=None, use_ddim=False, ddim_step=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        

        if not use_ddim:
            iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                            total=timesteps)

            for i in iterator:
                ts = torch.full((b,), i, device=self.device, dtype=torch.long)
                img, x0_partial = self.p_sample(img, cond, ts)
                if i % log_every_t == 0 or i == timesteps - 1:
                    intermediates.append(x0_partial)
        else:
            for i in tqdm(reversed(range(self.ddim_step-1, self.num_timesteps+1, self.ddim_step)), desc='DDIM Sampling ', total=int(timesteps/self.ddim_step)):
                ts = torch.full((b,), i, device=self.device, dtype=torch.long)
                img, x0 = self.p_sample_ddim(img, cond, ts)
                if (i+1) % log_every_t == 0 or i == self.ddim_step-1:
                    intermediates.append(x0)

        return img, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, return_keys=None,
                    plot_denoise_rows=True, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=2., unconditional_guidance_label=None,
                   **kwargs):

        log = dict()
        z, c, x, xrec= self.get_input(batch,return_first_stage_outputs=True,bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # log["inputs"] = x
        log["reconstruction"] = xrec



        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            
            # samples, z_denoise_row = self.sample_log(cond=c, batch_size=N)
            # x_samples = self.decode_first_stage(samples)
            # log["samples"] = x_samples
            # if plot_denoise_rows:
            #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            #     log["denoise_row"] = denoise_grid
            samples_ddim, z_denoise_row_ddim = self.sample_log(cond=c, batch_size=N, use_ddim = True)
            x_samples_ddim = (samples_ddim)
            log["ddim_samples"] = self.decode_first_stage(x_samples_ddim)
            if plot_denoise_rows:
                denoise_grid_ddim = self._get_denoise_row_from_list(z_denoise_row_ddim)
                log["ddim_denoise_row"] = denoise_grid_ddim

           

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            
            # samples_cfg, _ = self.sample_log(cond=c, batch_size=N,
            #                                     unconditional_guidance_scale=unconditional_guidance_scale,
            #                                     unconditional_conditioning=uc)
            # x_samples_cfg = samples_cfg
            # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            samples_cfg_ddim, _ = self.sample_log(cond=c, batch_size=N,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc, use_ddim = True)
            x_samples_cfg_ddim = self.decode_first_stage(samples_cfg_ddim)
            log[f"ddim_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg_ddim

        

        if plot_progressive_rows:
            # img, progressives = self.progressive_denoising(c,
            #                             shape=(self.channels, self.image_size, self.image_size), batch_size=N)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            # log["progressive_row"] = prog_row
            img_ddim, progressives_ddim = self.progressive_denoising(c,
                                        shape=(self.channels, self.image_size, self.image_size), batch_size=N, use_ddim=True)
            prog_row_ddim = self._get_denoise_row_from_list(progressives_ddim, desc="Progressive Generation")
            log["ddim_progressive_row"] = prog_row_ddim

        return log







############################################### optimizer ################################################
  

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        # params = params + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def training_epoch_end(self, outputs, *args, **kwargs):
        
        # avg_loss_epoch = torch.stack([x["loss"] for x in outputs]).mean()
        # print("avg_loss_epoch_training",avg_loss_epoch)
        return 
    
    
    

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")
            
            
            
    