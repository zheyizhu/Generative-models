# UNet Conditioning tests


```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --base configs/imagen.yaml -t --gpus 0,
```

```bash
skip_connection: True
implicit_cross_attention: True     # stable diffusion: False, Imagen: True,
explicit_cross_attention: False      # stable diffusion: True, Imagen: False,
cat_x_cond_sequence: False       
cat_x_cond_embedding: False      # stable diffusion: False, Imagen: False, stable diffusion xl: True
use_adding_condition: True        # stable diffusion: False, Imagen: True,
use_scale_shift_norm: True      # stable diffusion: False, Imagen: True
use_causal_mask: False
```

