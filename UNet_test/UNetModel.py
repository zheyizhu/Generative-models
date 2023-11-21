import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

from torchvision.utils import make_grid



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels = None, padding=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels = None, padding=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class TransposedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, ks = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.up = nn.ConvTranspose2d(self.in_channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64, dropout=0., 
                 implicit_cross_attention=False, explicit_cross_attention=False,
                 use_causal_mask = False):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.implicit_cross_attention = implicit_cross_attention 
        self.explicit_cross_attention = explicit_cross_attention

        inner_dim = dim_head * heads
        if context_dim == None:
            key_value_dim = query_dim
        else:
            key_value_dim = context_dim
        # elif self.implicit_cross_attention:
        #     key_value_dim = context_dim + query_dim
        # elif self.explicit_cross_attention:
        #     key_value_dim = context_dim


        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_value_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(key_value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # for self attention, q, k ,v = query, query, query
        if context is None:
            context = x
        if self.implicit_cross_attention or self.explicit_cross_attention:
            context = torch.reshape(context,(context.shape[0],1,context.shape[1]))

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # split dim for multi heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        if self.implicit_cross_attention:
            k = torch.cat((k,q), dim =-2)
            v = torch.cat((v,q), dim =-2)
            
        # compute similarity of q and k
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if self.use_causal_mask:
            causal_mask = torch.ones((q.shape[1],k.shape[1]), dtype=bool)
            if self.implicit_cross_attention:
                causal_mask = torch.tril(causal_mask, diagonal=k.shape[1]-q.shape[1])
            else:
                causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask.to(sim.device)
            causal_mask = repeat(causal_mask, 'i j -> (b) i j', b=sim.shape[0])
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~causal_mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        # attention * v
        out = einsum('b i j, b j d -> b i d', sim, v)

        # converge heads back
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, query_dim, n_heads, d_head, dropout=0., context_dim=None, implicit_cross_attention=False, explicit_cross_attention=False, use_causal_mask = False):
        super().__init__()
        self.implicit_cross_attention = implicit_cross_attention
        self.explicit_cross_attention = explicit_cross_attention

        # Imagen did text conditioning by using concat(cond, x) as key, value in self-attention
        if implicit_cross_attention:
          self.norm1 = nn.LayerNorm(query_dim)
          self.attn1 = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                context_dim=context_dim, implicit_cross_attention = self.implicit_cross_attention, 
                                use_causal_mask = use_causal_mask)  
        else:
          self.norm1 = nn.LayerNorm(query_dim)
          self.attn1 = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                context_dim=None, use_causal_mask = use_causal_mask)
    
        # stable diffusion: explicit cross attention = cond as key, value in self-attention
        if explicit_cross_attention: 
          self.norm2 = nn.LayerNorm(query_dim)
          self.attn2 = Attention(query_dim=query_dim, context_dim=context_dim,
                                heads=n_heads, dim_head=d_head, dropout=dropout, 
                                explicit_cross_attention = self.explicit_cross_attention)
        else:
          self.norm2 = None
          self.attn2 = None

        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, 4* query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4* query_dim, query_dim)
        )

    def forward(self, x, context=None):
        if self.implicit_cross_attention:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x), context=None) + x
        if self.explicit_cross_attention:
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
  


class SpatialTransformer(nn.Module):

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 implicit_cross_attention = False,
                 explicit_cross_attention=False,
                 cat_x_cond_sequence = False,
                 use_causal_mask = False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.cat_x_cond_sequence = cat_x_cond_sequence
        
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, 
                                implicit_cross_attention=implicit_cross_attention, 
                                explicit_cross_attention=explicit_cross_attention,
                                use_causal_mask = use_causal_mask)
                for d in range(depth)]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.proj_out.weight.data.zero_()
        self.proj_out.bias.data.zero_()

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        b, d = context.shape

        x_in = x
        
        x = self.norm(x)
        if self.cat_x_cond_sequence:
            # (bs,1,768) -> (bs,c,768)
            # context_prompt = repeat(context, 'b 1 d-> b (repeat 1) d', repeat=c)
            # (bs, 768) -> (bs,c,768)
            context_prompt = repeat(context, 'b d -> b c d', c=c)
            # (bs,c,h,w) -> bs,c,hw)
            x = torch.reshape(x, (b,c,h*w))
            x = torch.cat((context_prompt, x), dim=-1)
            x = rearrange(x, 'b c s -> b s c').contiguous()
        else:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
                

        x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context)

        x = self.proj_out(x)
        if self.cat_x_cond_sequence:
            x = x[:,d:,:]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        
        
        return x + x_in


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        dropout,
        cond_emb_channels,
        out_channels=None,
        use_scale_shift_norm=True, # stable diffusion: False, Imagen: True, 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        # input block
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)

        ### stable diffusion: time embedding, Imagen: concat(time, cond)
        self.cond_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                cond_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # output block
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.activation2 = nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(self, x, combined_cond_emb):
        h = x
        h = self.norm1(h)
        h = self.activation1(h)
        h = self.conv1(h)

        cond_emb_output = self.cond_embedding(combined_cond_emb).type(h.dtype)

        while len(cond_emb_output.shape) < len(h.shape):
            cond_emb_output = cond_emb_output[..., None]
          
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(cond_emb_output, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = h + cond_emb_output
            h = self.norm2(h)

        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        numResPerBlock,
        n_heads,
        time_emb_channels,
        stride=2,
        dropout=0.0,
        context_dim=0,
        cat_x_cond_sequence = False,  #prompting
        implicit_cross_attention = False,
        explicit_cross_attention = False,
        use_adding_condition = False,
        use_scale_shift_norm = True,
        use_causal_mask = False
    ):
        super().__init__()
        self.cat_x_cond_sequence = cat_x_cond_sequence
        self.implicit_cross_attention = implicit_cross_attention
        self.explicit_cross_attention = explicit_cross_attention
        self.use_adding_condition= use_adding_condition

        self.context_dim = context_dim

        self.down = Downsample(in_channels=in_channels)

        self.resnet_blocks = nn.ModuleList([ResNetBlock(in_channels, dropout, time_emb_channels+context_dim, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm) if self.use_adding_condition else
                                            ResNetBlock(in_channels, dropout, time_emb_channels, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm)]
                                           +[ResNetBlock(out_channels, dropout, time_emb_channels+context_dim, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm) if self.use_adding_condition else
                                            ResNetBlock(out_channels, dropout, time_emb_channels, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm) for d in range(numResPerBlock-1)])

        self.self_attn = SpatialTransformer(out_channels, n_heads=n_heads, d_head=2*out_channels//n_heads, depth=1, context_dim = context_dim,
                                            implicit_cross_attention = implicit_cross_attention, 
                                            explicit_cross_attention = explicit_cross_attention,
                                            cat_x_cond_sequence = cat_x_cond_sequence,
                                            use_causal_mask = use_causal_mask)        

    def forward(self, x, c = None, t = None, cond = None, use_downsample = False, use_self_attn = False):
        h = x

        if self.use_adding_condition:
            cond = torch.cat((t,c), dim=1)
        else:
            cond = t

        if use_downsample:
            h = self.down(h)
        
        for i, block in enumerate(self.resnet_blocks):
            h = block(h, cond)
        
        if use_self_attn:
            h = self.self_attn(h, c)

        return h

class UBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        numResPerBlock,
        n_heads,
        time_emb_channels,
        stride=2,
        dropout=0.0,
        context_dim=0,
        cat_x_cond_sequence = False, 
        implicit_cross_attention = False,
        explicit_cross_attention = False,
        use_adding_condition = False,
        use_scale_shift_norm = True,
        use_causal_mask = False
    ):
        super().__init__()
        self.cat_x_cond_sequence = cat_x_cond_sequence #prompting
        self.implicit_cross_attention = implicit_cross_attention
        self.explicit_cross_attention = explicit_cross_attention
        self.use_adding_condition= use_adding_condition
        
        self.context_dim = context_dim 

        self.resnet_blocks = nn.ModuleList([ResNetBlock(in_channels, dropout, time_emb_channels+context_dim, out_channels=out_channels, use_scale_shift_norm = use_scale_shift_norm) if self.use_adding_condition else
                                            ResNetBlock(in_channels, dropout, time_emb_channels, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm)]
                                           +[ResNetBlock(out_channels, dropout, time_emb_channels+context_dim, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm) if self.use_adding_condition else
                                            ResNetBlock(out_channels, dropout, time_emb_channels, out_channels=out_channels,use_scale_shift_norm = use_scale_shift_norm) for d in range(numResPerBlock-1)])

        self.self_attn = SpatialTransformer(out_channels, n_heads=n_heads, d_head=2*out_channels//n_heads, depth=1,context_dim = context_dim,
                                            implicit_cross_attention = implicit_cross_attention, 
                                            explicit_cross_attention = explicit_cross_attention,
                                            cat_x_cond_sequence = cat_x_cond_sequence,
                                            use_causal_mask = use_causal_mask)
        
        self.up = Upsample(in_channels=out_channels)
        

    def forward(self, x, c = None, t = None, cond = None, use_upsample = False, use_self_attn = False):
        h = x

        if self.use_adding_condition:
            cond = torch.cat((t,c), dim=1)
        else:
            cond = t

        for i, block in enumerate(self.resnet_blocks):
            h = block(h, cond)

        if use_self_attn:
            h = self.self_attn(h, c)

        if use_upsample:
            h = self.up(h)
        
        return h
    

class Imagen_UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        inner_channels=128,
        num_downsampling=2,
        numResPerBlock=3,
        n_heads=8,
        stride=2,
        dropout=0.0,
        context_dim=0,
        time_emb_channels=256,
        skip_connection=True,
        implicit_cross_attention = False, 
        explicit_cross_attention = False,
        cat_x_cond_embedding=False,
        cat_x_cond_sequence = False,
        use_adding_condition= False,
        use_scale_shift_norm = True,
        use_causal_mask = False,
    ):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.cat_x_cond_embedding = cat_x_cond_embedding
        remainder = context_dim % (input_size*input_size)
        cond_channel = context_dim // (input_size*input_size) + 1 if remainder!=0 else context_dim // (input_size*input_size)
        if self.cat_x_cond_embedding:
            in_channels += cond_channel
            out_channels += cond_channel
            
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_adding_condition = use_adding_condition
        self.skip_connection = skip_connection
        self.dtype = torch.float32
        self.inner_channels = inner_channels
        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(inner_channels, inner_channels*2),
            nn.SiLU(),
            nn.Linear(inner_channels*2, time_emb_channels),
        )

        self.conv_in = torch.nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1)

        self.DownBlocks = nn.ModuleList([DBlock(inner_channels, inner_channels, numResPerBlock, n_heads, 
                                                time_emb_channels, stride, dropout, context_dim,
                                                implicit_cross_attention = implicit_cross_attention, 
                                                explicit_cross_attention = explicit_cross_attention,
                                                cat_x_cond_sequence = cat_x_cond_sequence,
                                                use_adding_condition= use_adding_condition,
                                                use_scale_shift_norm = use_scale_shift_norm,
                                                use_causal_mask = use_causal_mask) 
                                                for d in range(num_downsampling+1)])
        
        self.MidBlocks = nn.ModuleList([ResNetBlock(inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition else 
                                        ResNetBlock(inner_channels, dropout, time_emb_channels, out_channels=inner_channels, use_scale_shift_norm = use_scale_shift_norm),
                                        SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1,context_dim = context_dim,
                                            implicit_cross_attention = implicit_cross_attention, 
                                            explicit_cross_attention = explicit_cross_attention,
                                            cat_x_cond_sequence = cat_x_cond_sequence,
                                            use_causal_mask = use_causal_mask),
                                        ResNetBlock(inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition else 
                                        ResNetBlock(inner_channels, dropout, time_emb_channels, out_channels=inner_channels, use_scale_shift_norm = use_scale_shift_norm)])
        
        if skip_connection:
            self.UpBlocks = nn.ModuleList([UBlock(inner_channels*2, inner_channels, numResPerBlock, n_heads, 
                                                  time_emb_channels, stride, dropout, context_dim,
                                                  implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_adding_condition= use_adding_condition,
                                                    use_scale_shift_norm = use_scale_shift_norm,
                                                    use_causal_mask = use_causal_mask) 
                                                    for d in range(num_downsampling+1)])
        else:
            self.UpBlocks = nn.ModuleList([UBlock(inner_channels, inner_channels, numResPerBlock, n_heads, 
                                                  time_emb_channels, stride, dropout, context_dim,
                                                  implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_adding_condition= use_adding_condition,
                                                    use_scale_shift_norm = use_scale_shift_norm,
                                                    use_causal_mask = use_causal_mask) 
                                                    for d in range(num_downsampling+1)])

        self.dense = nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1)

        

    def forward(self, x, t=None, c=None):
        bs, channel, height, width = x.shape
        if c is not None:
            assert c.shape[0] == x.shape[0]
            bs, context_dim = c.shape
        
        # c = (bs, 768), x = (bs, 64, 8, 8)
        if self.cat_x_cond_embedding:
            remainder = context_dim % (height*width)
            cond_channel = context_dim // (height*width) + 1 if remainder!=0 else context_dim // (height*width)
            pad_length = height*width-remainder if remainder!=0 else 0
            cond_padded = torch.nn.functional.pad(c, (0,pad_length),'constant',0)
            cond_reshaped = torch.reshape(cond_padded, (bs,cond_channel,height,width))
            x = torch.cat((cond_reshaped,x), dim=1)
            
            
        if c is not None:
            c = torch.reshape(c, (c.shape[0],-1))
        
        
        t_emb = repeat(t, 'b -> b d', d=self.inner_channels)
        t_emb = t_emb.type(self.dtype)
        time_emb = self.time_embed(t_emb)
    
        
        h = self.conv_in(x)

        hs = []
        for i, block in enumerate(self.DownBlocks):
            if i == 0:
                h = block(h, c, time_emb, use_downsample = False, use_self_attn = False)
            else:
                h = block(h, c, time_emb, use_downsample = True, use_self_attn = True)
            hs.append(h)
            
            
        for i, block in enumerate(self.MidBlocks):
            if isinstance(block, ResNetBlock):
                if self.use_adding_condition:
                    cond = torch.cat((time_emb,c), dim=1)
                else:
                    cond = time_emb
                h = block(h, cond)
            elif isinstance(block, SpatialTransformer):
                h = block(h, c)
            

        for i, block in enumerate(self.UpBlocks):
            if self.skip_connection:
                h = torch.cat([h, hs.pop()], dim=1)
            if i == len(self.UpBlocks)-1:
                h = block(h, c, time_emb, use_upsample = False, use_self_attn = False)
            else:
                h = block(h, c, time_emb, use_upsample = True, use_self_attn = True)
        
        h = self.dense(h)
                
        if self.cat_x_cond_embedding:
            h = h[:,cond_channel:,:,:]
        

        return h
    
    
class SD_UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        inner_channels=128,
        # num_downsampling=2,
        # numResPerBlock=3,
        n_heads=8,
        # stride=2,
        dropout=0.0,
        context_dim=0,
        time_emb_channels=256,
        skip_connection=True,
        implicit_cross_attention = False, 
        explicit_cross_attention = False,
        cat_x_cond_embedding=False,
        cat_x_cond_sequence = False,
        use_adding_condition= False,
        use_scale_shift_norm = True,
        use_causal_mask = False,                 
    ):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.cat_x_cond_embedding = cat_x_cond_embedding
        remainder = context_dim % (input_size*input_size)
        cond_channel = context_dim // (input_size*input_size) + 1 if remainder!=0 else context_dim // (input_size*input_size)
        if self.cat_x_cond_embedding:
            in_channels += cond_channel
            out_channels += cond_channel
            
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_adding_condition = use_adding_condition
        self.skip_connection = skip_connection
        self.dtype = torch.float32
        self.inner_channels = inner_channels
        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(inner_channels, inner_channels*2),
            nn.SiLU(),
            nn.Linear(inner_channels*2, time_emb_channels),
        )




        # down blocks
        # 64 -> 64
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.Conv2d(in_channels, inner_channels, 3, padding=1))
        # 64 -> 64
        
        
        self.down_blocks.append(ResNetBlock(inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(inner_channels, dropout, time_emb_channels, out_channels=inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.down_blocks.append(SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # 64 -> 64
        self.down_blocks.append(ResNetBlock(inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(inner_channels, dropout, time_emb_channels, out_channels=inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.down_blocks.append(SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # 64 -> 64
        self.down_blocks.append(Downsample(1 * inner_channels, 1 * inner_channels))
        # 64 -> 128
        self.down_blocks.append(ResNetBlock(inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.down_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # 128 -> 128
        self.down_blocks.append(ResNetBlock(2*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(2*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.down_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        
        # middle blocks
        self.mid_blocks = nn.ModuleList()
        self.mid_blocks.append(ResNetBlock(2*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(2*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.mid_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        self.mid_blocks.append(ResNetBlock(2*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(2*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))

        # up blocks
        # residuals from down blocks [64, 64, 64, 64, 128, 128]
        self.up_blocks = nn.ModuleList()
        # (128+128) -> 128
        self.up_blocks.append(ResNetBlock(4*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(4*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # (128+128) -> 128
        self.up_blocks.append(ResNetBlock(4*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(4*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # (128+64) -> 128
        self.up_blocks.append(ResNetBlock(3*inner_channels, dropout, time_emb_channels+context_dim, out_channels=2*inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(3*inner_channels, dropout, time_emb_channels, out_channels=2*inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(2*inner_channels, n_heads=n_heads, d_head=2*inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # upsample 128 -> 128
        self.up_blocks.append(TransposedUpsample(2 * inner_channels, 2 * inner_channels))
        # (128+64) -> 64
        self.up_blocks.append(ResNetBlock(3*inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(3*inner_channels, dropout, time_emb_channels, out_channels=inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # (64+64) -> 64
        self.up_blocks.append(ResNetBlock(2*inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(2*inner_channels, dropout, time_emb_channels, out_channels=inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))
        # (64+64) -> 64
        self.up_blocks.append(ResNetBlock(2*inner_channels, dropout, time_emb_channels+context_dim, out_channels=inner_channels, 
                                            use_scale_shift_norm = use_scale_shift_norm) if use_adding_condition 
                                else ResNetBlock(2*inner_channels, dropout, time_emb_channels, out_channels=inner_channels, 
                                                use_scale_shift_norm = use_scale_shift_norm))
        self.up_blocks.append(SpatialTransformer(inner_channels, n_heads=n_heads, d_head=inner_channels//n_heads, depth=1, context_dim = context_dim,
                                                    implicit_cross_attention = implicit_cross_attention, 
                                                    explicit_cross_attention = explicit_cross_attention,
                                                    cat_x_cond_sequence = cat_x_cond_sequence,
                                                    use_causal_mask = use_causal_mask))

        self.out = nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=inner_channels, eps=1e-6, affine=True),
            nn.SiLU(), 
            nn.Conv2d(inner_channels, out_channels, 3, padding=1)
            )

 

    def forward(self, x, t=None, c=None, **kwargs):
        bs, channel, height, width = x.shape
        if c is not None:
            assert c.shape[0] == x.shape[0]
            bs, context_dim = c.shape
        
        if self.cat_x_cond_embedding:
            remainder = context_dim % (height*width)
            cond_channel = context_dim // (height*width) + 1 if remainder!=0 else context_dim // (height*width)
            pad_length = height*width-remainder if remainder!=0 else 0
            cond_padded = torch.nn.functional.pad(c, (0,pad_length),'constant',0)
            cond_reshaped = torch.reshape(cond_padded, (bs,cond_channel,height,width))
            x = torch.cat((cond_reshaped,x), dim=1)
            
            
        if c is not None:
            c = torch.reshape(c, (c.shape[0],-1))
        
        
        t_emb = repeat(t, 'b -> b d', d=self.inner_channels)
        t_emb = t_emb.type(self.dtype)
        time_emb = self.time_embed(t_emb)
        
        

        h = x
                
                
        # down blocks
        hs = []
        for layer in self.down_blocks:
            if isinstance(layer, ResNetBlock):
                if self.use_adding_condition:
                    cond = torch.cat((time_emb,c), dim=1)
                else:
                    cond = time_emb
                h = layer(h, cond)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, c)
                hs.append(h)
            else:
                h = layer(h)
                hs.append(h)

        # mid blocks
        for layer in self.mid_blocks:
            if isinstance(layer, ResNetBlock):
                if self.use_adding_condition:
                    cond = torch.cat((time_emb,c), dim=1)
                else:
                    cond = time_emb
                h = layer(h, cond)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, c)
            else:
                h = layer(h)

        # up blocks
        for layer in self.up_blocks:
            if isinstance(layer, ResNetBlock):
                if self.skip_connection:
                    h = torch.cat([h, hs.pop()], dim=1)
                if self.use_adding_condition:
                    cond = torch.cat((time_emb,c), dim=1)
                else:
                    cond = time_emb
                h = layer(h, cond)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, c)
            else:
                h = layer(h)

        h = self.out(h)
        
        if self.cat_x_cond_embedding:
            h = h[:,cond_channel:,:,:]
            
        h = h.type(x.dtype)
        
        return h