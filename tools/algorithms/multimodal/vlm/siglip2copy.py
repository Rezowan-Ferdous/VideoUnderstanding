import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out


class Siglip2VisionOutput():
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Siglip2TextOutput():
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class Siglip2Output():
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Linear(

            in_features=config.num_channels * self.patch_size * self.patch_size,

            out_features=self.embed_dim,
        )


        self.num_patches = config.num_patches

        self.position_embedding_size = int(self.num_patches**0.5)

        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:


        batch_size = spatial_shapes.shape[0]

        embed_dim = positional_embeddings.shape[-1]

        source_dtype = positional_embeddings.dtype


        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )


        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # bfloat16/float16 的 antialias
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):

            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]

            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)


            resized_embeddings = resized_embeddings.to(source_dtype)


            resulted_positional_embeddings[i, : height * width] = resized_embeddings

            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """

        """

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )

        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_dim = config.hidden_size

        self.num_heads = config.num_attention_heads

        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, q_len, _ = hidden_states.size()


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale


        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )


        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Siglip2FlashAttention2(Siglip2Attention):
    """
    Siglip2Attention 的 Flash Attention 模块。该模块继承自 `Siglip2Attention`，因此模型的权重保持不变。
    唯一需要修改的是前向传播方法，需要正确调用 Flash Attention 的公共 API，并处理输入中可能存在的填充 token。

    Attributes:
        is_causal (bool): 
            是否为因果注意力。如果为 `True`，则只允许对当前和之前的 token 进行注意力计算。
    """

    is_causal = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播方法，计算 Flash Attention。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 (batch_size, 时间步长度, 通道数)。
            attention_mask (`torch.LongTensor`, 可选):
                注意力掩码张量，用于屏蔽某些位置，形状为 (batch_size, 1, 时间步长度, 时间步长度)。
            output_attentions (`bool`):
                是否输出注意力权重。

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`:
                - `attn_output`: 注意力输出，形状为 (batch_size, 时间步长度, 隐藏层大小)。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 (batch_size, num_heads, 时间步长度, 时间步长度)。
                - `attn_weights_tuple`: 其他注意力权重信息（可选）。
        """
        output_attentions = False

        # 获取批次大小和时间步长度
        batch_size, q_len, _ = hidden_states.size()

        # 计算查询 (query)、键 (key) 和值 (value) 的投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention 要求输入的形状为 (batch_size, seq_length, head_dim, hidden_dim)
        # 因此我们保持原始形状不变
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        dropout_rate = self.dropout if self.training else 0.0

        # 在 PEFT 中，通常我们将层归一化层转换为 float32 以提高训练稳定性
        # 因此，输入的隐藏状态会被静默地转换为 float32。因此，我们需要将其转换回正确的类型，以确保一切按预期工作。
        # 这种转换可能会减慢训练和推理速度，因此建议不要将 LayerNorms 转换为 fp32。

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # 处理模型量化的情形
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 调用 Flash Attention 的前向传播方法
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # 重塑注意力输出张量
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # 返回注意力输出和可选的注意力权重
        return attn_output, attn_weights


class Siglip2SdpaAttention(Siglip2Attention):
    """
    使用 torch.nn.functional.scaled_dot_product_attention 的 Siglip2 注意力模块。该模块继承自 `Siglip2Attention`，因为模块的权重保持不变。
    唯一的变化是在前向传播方法中，以适应 SDPA API。

    Attributes:
        is_causal (bool): 
            是否为因果注意力。如果为 `True`，则只允许对当前和之前的 token 进行注意力计算。
    """

    is_causal = False

    # Adapted from Siglip2Attention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播方法，计算使用 SDPA 的注意力。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 (batch_size, 时间步长度, 通道数)。
            attention_mask (`torch.Tensor`, 可选):
                注意力掩码张量，用于屏蔽某些位置，形状为 (batch_size, 1, 时间步长度, 时间步长度)。
            output_attentions (`bool`, 可选):
                是否输出注意力权重。

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor]]`:
                - `attn_output`: 注意力输出，形状为 (batch_size, 时间步长度, 隐藏层大小)。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 (batch_size, num_heads, 时间步长度, 时间步长度)。
        """
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        # 获取批次大小和时间步长度
        batch_size, q_len, _ = hidden_states.size()

        # 计算查询 (query)、键 (key) 和值 (value) 的投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑张量以适应多头注意力的计算
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # 我们通过 `is_causal` 语句而不是 SDPA 中的内联条件分配来调度到 SDPA 的 Flash Attention 或 Efficient 内核，
        # 以支持 torch.compile 的动态形状和完整图选项。内联条件会阻止动态形状的编译。
        is_causal = True if self.is_causal and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 重塑注意力输出张量
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        # 应用输出投影层
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出，不返回注意力权重
        return attn_output, None


class Siglip2MLP(nn.Module):
    """
    Siglip2 的多层感知机（MLP）模块，用于在注意力机制之后进行非线性变换。

    Args:
        config: 
            模型配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - intermediate_size (int): MLP 中间层的维度。
            - hidden_act (str): 激活函数的类型，如 "relu", "gelu" 等。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义第一个全连接层，将隐藏层大小映射到中间层大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 定义第二个全连接层，将中间层大小映射回隐藏层大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，应用 MLP 变换。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 `(batch_size, seq_length, hidden_size)`。

        Returns:
            `torch.Tensor`: 
                变换后的张量，形状为 `(batch_size, seq_length, hidden_size)`。
        """
        # 应用第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 返回变换后的张量
        return hidden_states


# 定义注意力机制的实现类映射
SIGLIP2_ATTENTION_CLASSES = {
    "eager": Siglip2Attention,
    "flash_attention_2": Siglip2FlashAttention2,
    "sdpa": Siglip2SdpaAttention,
}


class Siglip2EncoderLayer(nn.Module):
    """
    Siglip2 的编码器层，包含自注意力机制和 MLP 模块。

    Args:
        config (Siglip2Config): 
            模型配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - intermediate_size (int): MLP 中间层的维度。
            - hidden_act (str): 激活函数的类型。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - _attn_implementation (str): 注意力机制的实现方式，如 "eager", "flash_attention_2", "sdpa"。
    """
    def __init__(self, config: Siglip2Config):
        super().__init__()

        # 嵌入维度，通常与隐藏层大小相同
        self.embed_dim = config.hidden_size
        # 根据配置选择注意力机制的实现类，并实例化
        self.self_attn = SIGLIP2_ATTENTION_CLASSES[config._attn_implementation](config=config)
        # 定义第一个层归一化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 实例化 MLP 模块
        self.mlp = Siglip2MLP(config)
        # 定义第二个层归一化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        前向传播方法，处理输入张量通过自注意力和 MLP。

        Args:
            hidden_states (`torch.FloatTensor`):
                输入张量，形状为 `(batch_size, seq_len, embed_dim)`。
            attention_mask (`torch.FloatTensor`):
                注意力掩码张量，形状为 `(batch_size, 1, q_len, k_v_seq_len)`，其中填充元素由非常大的负值表示。
            output_attentions (`bool`, 可选, 默认值为 `False`):
                是否返回所有注意力层的注意力权重。

        Returns:
            `Tuple[torch.FloatTensor]`: 
                - `hidden_states`: 变换后的隐藏状态，形状为 `(batch_size, seq_len, embed_dim)`。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 `(batch_size, num_heads, q_len, k_v_seq_len)`。
        """
        # 保存残差连接的输入
        residual = hidden_states

        # 应用第一个层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 应用自注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
      
        hidden_states = residual + hidden_states

        
        residual = hidden_states

        
        hidden_states = self.layer_norm2(hidden_states)
       
        hidden_states = self.mlp(hidden_states)
       
        hidden_states = residual + hidden_states
      
        outputs = (hidden_states,)

        if output_attentions:
            
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        
        hidden_states = inputs_embeds

       
        for encoder_layer in self.layers:
            if output_hidden_states:
                
                encoder_states = encoder_states + (hidden_states,)

            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
           
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )