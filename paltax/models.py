# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    skip_traditional: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if self.skip_traditional:
            # Use the traditional ResNet skip connection
            if residual.shape != y.shape:
                residual = self.conv(self.filters * 4, (1, 1),
                                    self.strides, name='conv_proj')(residual)
                residual = self.norm(name='norm_proj')(residual)
        else:
            # Use the ResNetD skip connection.
            if self.strides != (1,1):
                residual = nn.avg_pool(residual, (2,2), strides=(2,2),
                                       padding=((0, 0), (0,0)))
            if residual.shape != y.shape:
                residual = self.conv(self.filters * 4, (1, 1), (1, 1),
                                     name='conv_proj')(residual)
                residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetDBlock(BottleneckResNetBlock):
    "Bottleneck ResNetD block."
    skip_traditional: bool = False


class ResNet(nn.Module):
    """ResNet Class"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_outputs: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm, use_running_average=not train,
            momentum=0.9, epsilon=1e-5, dtype=self.dtype
        )

        x = conv(self.num_filters, (7, 7), (2, 2),
                         padding=[(3, 3), (3, 3)],
                         name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i, strides=strides,
                    conv=conv, norm=norm, act=self.act
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_outputs, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


class ResNetD(nn.Module):
    """ResNet Class"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_outputs: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm, use_running_average=not train,
            momentum=0.9, epsilon=1e-5, dtype=self.dtype
        )

        # First stem.
        x = conv(self.num_filters // 2, (3, 3), (2, 2),
                 padding=[(1, 1), (1, 1)], name='conv_init_1')(x)
        x = norm(name='bn_init_1')(x)
        x = nn.relu(x)

        # Second stem.
        x = conv(self.num_filters // 2, (3, 3), (1, 1),
                 padding=[(1, 1), (1, 1)], name='conv_init_2')(x)
        x = norm(name='bn_init_2')(x)
        x = nn.relu(x)

        # Third stem.
        x = conv(self.num_filters, (3, 3), (1, 1),
                 padding=[(1, 1), (1, 1)], name='conv_init_3')(x)
        x = norm(name='bn_init_3')(x)
        x = nn.relu(x)

        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i, strides=strides,
                    conv=conv, norm=norm, act=self.act
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_outputs, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18VerySmall = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                            block_cls=ResNetBlock, num_filters=8)
ResNet18Small = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, num_filters=16)
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)

ResNetD50 = partial(ResNetD, stage_sizes=[3, 4, 6, 3],
                    block_cls=BottleneckResNetDBlock)

ResNet18Local = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)


# ===========================================================================
# =                                                                         =
# = Models added to Paltax by Koen Waggeveld                                =
# = Source: https://github.com/google-research/vision_transformer/tree/main =
# =                                                                         =
# ===========================================================================


from paltax.vit_jax.models_vit import VisionTransformer
import ml_collections

# vit_jax/configs/models.py: get_ti16_config()
transformer_ti = ml_collections.ConfigDict()
transformer_ti.mlp_dim = 768
transformer_ti.num_heads = 3
transformer_ti.num_layers = 12
transformer_ti.attention_dropout_rate = 0.0
transformer_ti.dropout_rate = 0.0

# vit_jax/configs/models.py: get_s16_config()
transformer_s = ml_collections.ConfigDict()
transformer_s.mlp_dim = 1536
transformer_s.num_heads = 6
transformer_s.num_layers = 12
transformer_s.attention_dropout_rate = 0.0
transformer_s.dropout_rate = 0.0

# vit_jax/configs/models.py: get_b16_config()
transformer_b = ml_collections.ConfigDict()
transformer_b.mlp_dim = 3072
transformer_b.num_heads = 12
transformer_b.num_layers = 12
transformer_b.attention_dropout_rate = 0.0
transformer_b.dropout_rate = 0.0

# vit_jax/configs/models.py: get_b16_config()
transformer_l = ml_collections.ConfigDict()
transformer_l.mlp_dim = 4096
transformer_l.num_heads = 16
transformer_l.num_layers = 24
transformer_l.attention_dropout_rate = 0.0
transformer_l.dropout_rate = 0.1


# ==========
# = ViT_Ti =
# ==========

ViT_Ti = partial(VisionTransformer,
            num_classes = 0,                    # Skips final if-statement
            transformer = transformer_ti,
            hidden_size = 192,
            classifier = 'token'
         )

ViT_Ti8  = partial(ViT_Ti, 
                   model_name = 'ViT-Ti_8',  
                   patches = ml_collections.ConfigDict({'size': (8, 8)}),
           )
ViT_Ti16 = partial(ViT_Ti, 
                   model_name = 'ViT-Ti_16', 
                   patches = ml_collections.ConfigDict({'size': (16, 16)}),
           )
ViT_Ti32 = partial(ViT_Ti, 
                   model_name = 'ViT-Ti_32', 
                   patches = ml_collections.ConfigDict({'size': (32, 32)}),
           )



# =============================================================
# = ViT_S: has similar number of parameters to ResNet50 above =
# =============================================================

ViT_S = partial(VisionTransformer,
            num_classes = 0,                    # Skips final if-statement
            transformer = transformer_s,
            hidden_size = 384,
            classifier = 'token'
        )   

ViT_S8  = partial(ViT_S, 
                  model_name = 'ViT-S_8',  
                  patches = ml_collections.ConfigDict({'size': (8, 8)}),
          )
ViT_S16  = partial(ViT_S, 
                  model_name = 'ViT-S_16',  
                  patches = ml_collections.ConfigDict({'size': (16, 16)}),
          )
ViT_S32  = partial(ViT_S, 
                  model_name = 'ViT-S_32',  
                  patches = ml_collections.ConfigDict({'size': (32, 32)}),
          )


# =========
# = ViT_L =
# =========

ViT_L16 = partial(VisionTransformer,            # Larger transformer (?)
            model_name = 'ViT-L_16',
            num_classes = 0,                    # Skips final if-statement
            patches = ml_collections.ConfigDict({'size': (16, 16)}),
            transformer = transformer_l,
            hidden_size = 1024,
            classifier = 'token'
            )


# ===========
# = Hybrids =
# ===========

# vit_jax/configs/models.py: get_r26_s32_config()
r26 = ml_collections.ConfigDict()
r26.num_layers = (2, 2, 2, 2)         # Using four bottleneck blocks results in a downscaling of 2^(1 + 4)=32 which
r26.width_factor = 1                  # results in an effective patch size of /32.

r50 = ml_collections.ConfigDict()     # Note that the "real" Resnet50 has (3, 4, 6, 3) bottleneck blocks. Here
r50.num_layers = (3, 4, 9)            # we're using (3, 4, 9) configuration so we get a downscaling of 2^(1 + 3)=16
r50.width_factor = 1                  # which results in an effective patch size of /16.


R26_S32 = partial(VisionTransformer,
            model_name = 'R26+ViT-S_32',
            num_classes = 0,                    # Skips final if-statement
            patches = ml_collections.ConfigDict({'size': (1, 1)}),
            transformer = transformer_s,
            hidden_size = 384,
            classifier = 'token',
            resnet = r26
            )
