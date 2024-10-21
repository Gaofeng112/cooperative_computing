import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from .layers import GEGLU, PaddedConv2D, apply_seq, apply_seq_lcm, td_dot


class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels, idx=None, optimization=False):
        super().__init__()
        self.idx = idx
        if self.idx is not None and optimization:
            with open('res_clip_value.json', 'r') as f:
                clip_value = json.load(f)
            if self.idx > 0:
                with open('conv_shortcut_clip_value.json', 'r') as f:
                    conv_shorcut_clip_value = json.load(f)
            self.in_layers = [ 
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.activations.swish,
                keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[self.idx][0], clip_value_max=clip_value[self.idx][0])),
                PaddedConv2D(out_channels, 3, padding=1),
            ]
            self.emb_layers = [
                keras.activations.swish,
                keras.layers.Dense(out_channels),
            ]
            self.out_layers = [
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.activations.swish,
                keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[self.idx][1], clip_value_max=clip_value[self.idx][1])),
                PaddedConv2D(out_channels, 3, padding=1),
            ]
            if idx > 0:
                self.clip_skip_connection = keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-conv_shorcut_clip_value[idx-1][0], clip_value_max=conv_shorcut_clip_value[idx-1][0]))
            else:
                self.clip_skip_connection = lambda x: x
            self.skip_connection = (
                PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
            )
        else:
            self.in_layers = [
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.activations.swish,
                PaddedConv2D(out_channels, 3, padding=1),
            ]
            self.emb_layers = [
                keras.activations.swish,
                keras.layers.Dense(out_channels),
            ]
            self.out_layers = [
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.activations.swish,
                PaddedConv2D(out_channels, 3, padding=1),
            ]
            self.clip_skip_connection = lambda x: x
            self.skip_connection = (
                PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
            )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.clip_skip_connection(self.skip_connection(x)) + h
        return ret


class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_head, idx=None, optimization=False):
        super().__init__()
        self.idx = idx
        if self.idx is not None and optimization:
            with open('attn_clip_value.json', 'r') as f:
                clip_value = json.load(f)
            self.clip_value = clip_value[idx]
        else:
            self.clip_value = None
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head**-0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [keras.layers.Dense(n_heads * d_head)]

    def call(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        if self.clip_value is not None:
            q, k, v = (
                self.to_q(tf.clip_by_value(x,clip_value_min=-self.clip_value[0],clip_value_max=self.clip_value[0])), 
                self.to_k(tf.clip_by_value(context,clip_value_min=-self.clip_value[1],clip_value_max=self.clip_value[1])), 
                self.to_v(tf.clip_by_value(context,clip_value_min=-self.clip_value[2],clip_value_max=self.clip_value[2]))
            )
        else:
            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = keras.layers.Permute((2, 1, 3))(q)  # (bs, num_heads, time, head_size)
        k = keras.layers.Permute((2, 3, 1))(k)  # (bs, num_heads, head_size, time)
        v = keras.layers.Permute((2, 1, 3))(v)  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attention = td_dot(weights, v)
        attention = keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, num_heads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        if self.clip_value is not None:
            h_ = tf.clip_by_value(h_,clip_value_min=-self.clip_value[3],clip_value_max=self.clip_value[3])
        return apply_seq(h_, self.to_out)


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head, idx=None, optimization=False):
        super().__init__()
        self.ff_clip_value = None
        self.idx = idx
        self.optimization = optimization
        if idx is not None:
            self.idx = idx // 2
        if self.idx is not None and optimization:
            with open('ff_clip_value.json', 'r') as f:
                self.ff_clip_value = json.load(f)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head, idx=idx)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head, idx=(idx+1) if idx is not None else None)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        x, context = inputs
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        h = self.norm3(x)
        if self.ff_clip_value is not None and self.optimization:
            h = tf.clip_by_value(h, clip_value_min=-self.ff_clip_value[self.idx][0], clip_value_max=self.ff_clip_value[self.idx][0])
            h = self.geglu(h)
            h = tf.clip_by_value(h, clip_value_min=-self.ff_clip_value[self.idx][1], clip_value_max=self.ff_clip_value[self.idx][1])
            return self.dense(h) + x
        else:
            return self.dense(self.geglu(h)) + x


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels, n_heads, d_head, idx=None, optimization=False):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.idx = idx
        self.optimizaiton = optimization
        assert channels == n_heads * d_head
        self.proj_in = PaddedConv2D(n_heads * d_head, 1)
        self.proj_in_clip,self.proj_out_clip = None,None
        if self.idx is not None and optimization:
            with open("proj_clip_value.json","r") as fp:
                clip_value = json.load(fp)
            self.proj_in_clip = keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[idx][0], clip_value_max=clip_value[idx][0]))
            self.proj_out_clip = keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[idx][1], clip_value_max=clip_value[idx][1]))
            self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head, idx=self.idx*2, optimization=optimization)]
        else:
            self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head, idx=None, optimization=optimization)]
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, inputs):
        x, context = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        if self.proj_in_clip is not None and self.optimizaiton:
            x = self.proj_in_clip(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        if self.proj_out_clip is not None and self.optimizaiton:
            x = self.proj_out_clip(x)
        return self.proj_out(x) + x_in


class Downsample(keras.layers.Layer):
    def __init__(self, channels, idx=None, optimization=False):
        super().__init__()
        self.clip_value = None
        self.idx = idx
        self.optimization = optimization
        if self.idx is not None and optimization:
            with open('down_sample_clip_value.json', 'r') as f:
                self.clip_value = json.load(f)
            
        self.op = PaddedConv2D(channels, 3, stride=2, padding=1)

    def call(self, x):
        if self.clip_value is not None and self.optimization:
            return tf.clip_by_value(self.op(x), clip_value_min=-self.clip_value[self.idx][0], clip_value_max=self.clip_value[self.idx][0])
        else:
            return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, idx=None, optimization=False):
        super().__init__()
        self.clip_value = None
        self.idx = idx
        self.optimization = optimization
        if self.idx is not None:
            with open('up_sample_clip_value.json', 'r') as f:
                self.clip_value = json.load(f)
            
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, x):
        if self.clip_value is not None and self.optimization:
            x = tf.clip_by_value(x, clip_value_min=-self.clip_value[self.idx][0], clip_value_max=self.clip_value[self.idx][0])
        x = self.ups(x)
        return self.conv(x)


class UNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320), SpatialTransformer(320, 8, 40)],
            [ResBlock(320, 320), SpatialTransformer(320, 8, 40)],
            [Downsample(320)],
            [ResBlock(320, 640), SpatialTransformer(640, 8, 80)],
            [ResBlock(640, 640), SpatialTransformer(640, 8, 80)],
            [Downsample(640)],
            [ResBlock(640, 1280), SpatialTransformer(1280, 8, 160)],
            [ResBlock(1280, 1280), SpatialTransformer(1280, 8, 160)],
            [Downsample(1280)],
            [ResBlock(1280, 1280)],
            [ResBlock(1280, 1280)],
        ]
        self.middle_block = [
            ResBlock(1280, 1280),
            SpatialTransformer(1280, 8, 160),
            ResBlock(1280, 1280),
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280)],
            [ResBlock(2560, 1280)],
            [ResBlock(2560, 1280), Upsample(1280)],
            [ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)],
            [ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)],
            [
                ResBlock(1920, 1280),
                SpatialTransformer(1280, 8, 160),
                Upsample(1280),
            ],
            [ResBlock(1920, 640), SpatialTransformer(640, 8, 80)],  # 6
            [ResBlock(1280, 640), SpatialTransformer(640, 8, 80)],
            [
                ResBlock(960, 640),
                SpatialTransformer(640, 8, 80),
                Upsample(640),
            ],
            [ResBlock(960, 320), SpatialTransformer(320, 8, 40)],
            [ResBlock(640, 320), SpatialTransformer(320, 8, 40)],
            [ResBlock(640, 320), SpatialTransformer(320, 8, 40)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb, context = inputs
        # print("the t_emb is: ", t_emb)
        emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)

        for layer in self.middle_block:
            x = apply(x, layer)

        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                x = apply(x, layer)
        return apply_seq(x, self.out)

class BkUNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320), SpatialTransformer(320, 8, 40)],
            [Downsample(320)],
            [ResBlock(320, 640), SpatialTransformer(640, 8, 80)],
            [Downsample(640)],
            [ResBlock(640, 1280), SpatialTransformer(1280, 8, 160)],
            [Downsample(1280)],
            [ResBlock(1280, 1280)],
        ]
        self.middle_block = [
            ResBlock(1280, 1280),
            SpatialTransformer(1280, 8, 160),
            ResBlock(1280, 1280),
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280)],
            [ResBlock(2560, 1280), Upsample(1280)],
            [ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)],
            [
                ResBlock(1920, 1280),
                SpatialTransformer(1280, 8, 160),
                Upsample(1280),
            ],
            [ResBlock(1920, 640), SpatialTransformer(640, 8, 80)],  # 6
            [
                ResBlock(960, 640),
                SpatialTransformer(640, 8, 80),
                Upsample(640),
            ],
            [ResBlock(960, 320), SpatialTransformer(320, 8, 40)],
            [ResBlock(640, 320), SpatialTransformer(320, 8, 40)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb, context = inputs
        # print("the t_emb is: ", t_emb)
        emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        # print("="*20 , "input_blocks", "="*20)
        for b in self.input_blocks:
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
            saved_inputs.append(x)
            
        # print("="*20 , "middle_block", "="*20)
        for layer in self.middle_block:
            # print(layer.name)
            x = apply(x, layer)
            
        # print("="*20 , "output_blocks", "="*20)
        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
        return apply_seq(x, self.out)

class BkTinyUNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320), SpatialTransformer(320, 8, 40)],
            [Downsample(320)],
            [ResBlock(320, 640), SpatialTransformer(640, 8, 80)],
            [Downsample(640)],
            [ResBlock(640, 1280), SpatialTransformer(1280, 8, 160)],
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)],
            [
                ResBlock(1920, 1280),
                SpatialTransformer(1280, 8, 160),
                Upsample(1280),
            ],
            [ResBlock(1920, 640), SpatialTransformer(640, 8, 80)],  # 6
            [
                ResBlock(960, 640),
                SpatialTransformer(640, 8, 80),
                Upsample(640),
            ],
            [ResBlock(960, 320), SpatialTransformer(320, 8, 40)],
            [ResBlock(640, 320), SpatialTransformer(320, 8, 40)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb, context = inputs
        # print("the t_emb is: ", t_emb)
        emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        # print("="*20 , "input_blocks", "="*20)
        for b in self.input_blocks:
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
            saved_inputs.append(x)
            
        # print("="*20 , "output_blocks", "="*20)
        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
        return apply_seq(x, self.out)

class BkTinyLCMUNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.layers.Dense(320, use_bias=False),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320, idx=0), SpatialTransformer(320, 8, 40, idx=0)],
            [Downsample(320, idx=0)],
            [ResBlock(320, 640, idx=1), SpatialTransformer(640, 8, 80, idx=1)],
            [Downsample(640, idx=1)],
            [ResBlock(640, 1280, idx=2), SpatialTransformer(1280, 8, 160, idx=2)],
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280, idx=3), SpatialTransformer(1280, 8, 160, idx=3)],
            [
                ResBlock(1920, 1280, idx=4),
                SpatialTransformer(1280, 8, 160, idx=4),
                Upsample(1280, idx=0),
            ],
            [ResBlock(1920, 640, idx=5), SpatialTransformer(640, 8, 80, idx=5)],  # 6
            [
                ResBlock(960, 640, idx=6),
                SpatialTransformer(640, 8, 80, idx=6),
                Upsample(640, idx=1),
            ],
            [ResBlock(960, 320, idx=7), SpatialTransformer(320, 8, 40, idx=7)],
            [ResBlock(640, 320, idx=8), SpatialTransformer(320, 8, 40, idx=8)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb, t_cond, context = inputs
        # print("the t_emb is: ", t_emb)
        emb = apply_seq_lcm(t_emb, t_cond, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        # print("="*20 , "input_blocks", "="*20)
        for b in self.input_blocks:
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
            saved_inputs.append(x)
            
        # print("="*20 , "output_blocks", "="*20)
        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            # print(tf.reduce_mean(tf.abs(x[:,:,:,:320])))
            # print(tf.reduce_mean(tf.abs(x[:,:,:,320:])))
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
        return apply_seq(x, self.out)

class SplitResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels, idx=None):
        super().__init__()
        self.idx = None
        if self.idx is not None:
            with open('/megadisk/kris/tflite_deploy/bk-sdm-tflite-inference/conversion/stable_diffusion_tf/clip_value.json', 'r') as f:
                clip_value = json.load(f)
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ] if idx is None else [tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[idx][0], clip_value_max=clip_value[idx][0])),
            PaddedConv2D(out_channels, 3, padding=1),]
        
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(out_channels),
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ] if idx is None else [tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-clip_value[idx][1], clip_value_max=clip_value[idx][1])),
            PaddedConv2D(out_channels, 3, padding=1),]
        self.skip_connection_1 = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )
        self.skip_connection_2 = (
            PaddedConv2D(out_channels, 1, use_bias=False) if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x, y, emb = inputs
        h = apply_seq(tf.concat([x, y], axis=-1), self.in_layers)
        # intermediat_value_dict_split["in_layers_h"] = h
        emb_out = apply_seq(emb, self.emb_layers)
        # intermediat_value_dict_split["emb_out"] = emb_out
        h = h + emb_out[:, None, None]
        # intermediat_value_dict_split["h_emb_out"] = h
        h = apply_seq(h, self.out_layers)
        # intermediat_value_dict_split["out_layers_h"] = h
        ret1 = self.skip_connection_1(x)
        ret2 = self.skip_connection_2(y)
        return ret1 + ret2 + h

class BkTinyLCMSplitUNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.layers.Dense(320, use_bias=False),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320, idx=0), SpatialTransformer(320, 8, 40, idx=0)],
            [Downsample(320)],
            [ResBlock(320, 640, idx=1), SpatialTransformer(640, 8, 80, idx=1)],
            [Downsample(640)],
            [ResBlock(640, 1280, idx=2), SpatialTransformer(1280, 8, 160, idx=2)],
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280, idx=3), SpatialTransformer(1280, 8, 160, idx=3)],
            [
                ResBlock(1920, 1280, idx=4),
                SpatialTransformer(1280, 8, 160, idx=4),
                Upsample(1280),
            ],
            [ResBlock(1920, 640, idx=5), SpatialTransformer(640, 8, 80, idx=5)],  # 6
            [
                ResBlock(960, 640, idx=6),
                SpatialTransformer(640, 8, 80, idx=6),
                Upsample(640),
            ],
            [SplitResBlock(960, 320, idx=7), SpatialTransformer(320, 8, 40, idx=7)],
            [SplitResBlock(640, 320, idx=8), SpatialTransformer(320, 8, 40, idx=8)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb, t_cond, context = inputs
        # print("the t_emb is: ", t_emb)
        emb = apply_seq_lcm(t_emb, t_cond, self.time_embed)
        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)
            
        # global register_value_x_split, register_value_y_split
        for b in self.output_blocks:
            y = saved_inputs.pop()
            for layer in b:
                if isinstance(layer, SplitResBlock):
                    # register_value_y_split.append(y)
                    # register_value_x_split.append(x)
                    x = layer([x, y, emb])
                elif isinstance(layer, ResBlock):
                    x = layer([tf.concat([x, y], axis=-1), emb])
                elif isinstance(layer, SpatialTransformer):
                    x = layer([x, context])
                else:
                    x = layer(x)
        return apply_seq(x, self.out)


class BkTinyLCMUNetModelAll(keras.models.Model):
    def __init__(self, data, optimization=False):
        super().__init__()
        res_idx = 0
        attn_idx = 0
        d_sample_idx = 0
        u_sample_idx = 0
        self.t_cond_dim = 0
        if 'time_cond_proj_dim' in data and data.get('time_cond_proj_dim'):
            self.t_cond_dim = data.get('time_cond_proj_dim')
            self.time_embed = [
                keras.layers.Dense(1280),
                keras.layers.Dense(320, use_bias=False),
                keras.activations.swish,
                keras.layers.Dense(1280),
            ]
        else:
            self.time_embed = [
                keras.layers.Dense(1280),
                keras.activations.swish,
                keras.layers.Dense(1280),
            ]
        block_out_channels = data.get('block_out_channels')
        down_block_types = data.get('down_block_types')
        layers_per_block = data.get('layers_per_block')
        last_dim = block_out_channels[0]
        shortcut_block_channels = []
        shortcut_block_channels.append(block_out_channels[0]) 

        # down(input) blocks
        self.input_blocks = [[PaddedConv2D(block_out_channels[0], kernel_size=3, padding=1)],]
        for i, name in enumerate(down_block_types):
            if name == 'DownBlock2D':
                for j in range(layers_per_block):
                    res_in_channels = last_dim if j == 0 else block_out_channels[i]
                    self.input_blocks.append([
                        ResBlock(res_in_channels, block_out_channels[i], idx=res_idx, optimization=optimization),
                    ])
                    res_idx += 1
                    shortcut_block_channels.append(block_out_channels[i])
            elif name == "CrossAttnDownBlock2D":
                for j in range(layers_per_block):
                    res_in_channels = last_dim if j == 0 else block_out_channels[i]
                    self.input_blocks.append([
                        ResBlock(res_in_channels, block_out_channels[i], idx=res_idx, optimization=optimization),
                        SpatialTransformer(block_out_channels[i], 8, block_out_channels[i]//8, idx=attn_idx, optimization=optimization)
                    ])
                    res_idx += 1
                    attn_idx += 1
                    shortcut_block_channels.append(block_out_channels[i])
            if i != len(down_block_types) - 1:
                self.input_blocks.append([Downsample(block_out_channels[i], idx=d_sample_idx, optimization=optimization)])
                d_sample_idx += 1
                shortcut_block_channels.append(block_out_channels[i])
            last_dim = block_out_channels[i]

        # mid blocks
        self.mid_blocks = []
        if data.get('mid_block_type') == 'UNetMidBlock2DCrossAttn':
            self.mid_blocks.append([
                ResBlock(last_dim, last_dim, idx=res_idx, optimization=optimization),
                SpatialTransformer(last_dim, 8, last_dim//8, idx=attn_idx, optimization=optimization),
                ResBlock(last_dim, last_dim, idx=res_idx+1, optimization=optimization),
            ])
            res_idx += 2
            attn_idx += 1
        
        # up(output) blocks
        up_block_types = data.get('up_block_types')
        last_dim = block_out_channels[-1]
        self.output_blocks = []
        reversed_shortcut_block_channels = shortcut_block_channels[::-1]
        reversed_block_out_channels = block_out_channels[::-1]
        k = 0
        for i, name in enumerate(up_block_types):
            channels_num = reversed_block_out_channels[i] 
            if name == 'UpBlock2D':
                # 这里需要 +1 以保证shortcut通道的一致
                for j in range(layers_per_block + 1):
                    res_in_channels = (last_dim if j == 0 else channels_num) + reversed_shortcut_block_channels[k]
                    self.output_blocks.append([
                        ResBlock(res_in_channels, channels_num, idx=res_idx, optimization=optimization),
                    ])
                    res_idx += 1
                    k += 1
            elif name == "CrossAttnUpBlock2D":
                # 这里需要 +1 以保证shortcut通道的一致
                for j in range(layers_per_block + 1):
                    res_in_channels = (last_dim if j == 0 else channels_num) + reversed_shortcut_block_channels[k]
                    self.output_blocks.append([
                        ResBlock(res_in_channels, channels_num, idx=res_idx, optimization=optimization),
                        SpatialTransformer(channels_num, 8, channels_num//8, idx=attn_idx, optimization=optimization)
                    ])
                    res_idx += 1
                    attn_idx += 1
                    k += 1
            if i != len(up_block_types) - 1:
                self.output_blocks[-1].append(Upsample(channels_num, idx=u_sample_idx, optimization=optimization))
                u_sample_idx += 1
        
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def get_guidance_scale_embedding(self, w, embedding_dim=512):
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        
        if embedding_dim % 2 == 1:  # zero pad
            emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
        
        return emb
    def call(self, inputs):
        x, t_emb, context = inputs
        # print("the t_emb is: ", t_emb)
        if self.t_cond_dim:
            t_cond = self.get_guidance_scale_embedding(7.5, self.t_cond_dim)
            emb = apply_seq_lcm(t_emb, t_cond, self.time_embed)
        else:
            emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        # print("="*20 , "input_blocks", "="*20)
        for b in self.input_blocks:
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
            saved_inputs.append(x)
        
        if self.mid_blocks:
            for layer in self.mid_blocks[0]:
                x = apply(x, layer)
            
        # print("="*20 , "output_blocks", "="*20)
        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            # print(tf.reduce_mean(tf.abs(x[:,:,:,:320])))
            # print(tf.reduce_mean(tf.abs(x[:,:,:,320:])))
            for layer in b:
                # print(layer.name)
                x = apply(x, layer)
        return apply_seq(x, self.out)


