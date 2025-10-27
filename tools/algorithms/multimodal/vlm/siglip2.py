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

# Assuming configuration_siglip2.py exists in the same directory
# from configuration_siglip2 import Siglip2Config, Siglip2TextConfig, Siglip2VisionConfig

# Placeholder for configuration classes if the file is not available
class Siglip2Config:
    pass
class Siglip2TextConfig:
    pass
class Siglip2VisionConfig:
    pass

# Placeholder for Hugging Face PreTrainedModel and outputs if not imported
class PreTrainedModel(nn.Module):
    config_class = Siglip2Config
    base_model_prefix = "siglip2"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        self._init_weights(self)

    def _init_weights(self, module):
        pass

    @classmethod
    def _from_config(cls, config):
        return cls(config)

@dataclass
class BaseModelOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    def to_tuple(self):
        return tuple(
            getattr(self, k) for k in self.keys()
        )

    def keys(self):
        return [k for k in self.__annotations__ if getattr(self, k) is not None]
    
    def __getitem__(self, k):
        return getattr(self, k)

@dataclass
class BaseModelOutputWithPooling(BaseModelOutput):
    pooler_output: torch.FloatTensor = None

@dataclass
class ImageClassifierOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

# Placeholder for ACT2FN if not imported from transformers.activations
ACT2FN = {"gelu": nn.GELU(), "relu": nn.ReLU()}

# Placeholder for logging
import logging
logger = logging.getLogger(__name__)

# Placeholder for Flash Attention
def _flash_attention_forward(*args, **kwargs):
    raise ImportError("Flash Attention 2 is not available. Please install it.")
    
def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    """Creates a 4D attention mask from a 2D mask."""
    bsz, src_len = mask.shape
    if tgt_len is None:
        tgt_len = src_len
    
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    
    # Invert mask: 0s become -inf, 1s become 0s
    inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    
    return inverted_mask


@dataclass
class Siglip2VisionOutput():
    """
    Base class for vision model outputs, containing image embeddings obtained from pooling
    the last hidden state.

    Args:
        image_embeds (torch.FloatTensor, optional, shape `(batch_size, output_dim)`):
            Image embeddings obtained by applying the projection layer to `pooler_output`.
            Returned when `with_projection=True` is set during model initialization.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, output_dim)`
            - **Description**: Image embeddings representing the image features.
        
        last_hidden_state (torch.FloatTensor, required, shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden states at the output of the last layer of the model.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, sequence_length, hidden_size)`
            - **Description**: Contains the hidden states from the last layer, used for subsequent tasks.
        
        hidden_states (tuple(torch.FloatTensor), optional):
            Returned when `output_hidden_states=True` is passed or `config.output_hidden_states=True`.
            - **Type**: `tuple(torch.FloatTensor)`
            - **Description**: Tuple of hidden states for each layer, plus the initial embedding output.
            - **Shape**: `(batch_size, sequence_length, hidden_size)` for each tuple element.
        
        attentions (tuple(torch.FloatTensor), optional):
            Returned when `output_attentions=True` is passed or `config.output_attentions=True`.
            - **Type**: `tuple(torch.FloatTensor)`
            - **Description**: Tuple of attention weights after softmax, used to compute the weighted average in self-attention heads.
            - **Shape**: `(batch_size, num_heads, sequence_length, sequence_length)` for each tuple element.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Siglip2TextOutput():
    """
    Base class for text model outputs, containing text embeddings obtained from pooling
    the last hidden state.

    Args:
        text_embeds (torch.FloatTensor, optional, shape `(batch_size, output_dim)`):
            Text embeddings obtained by applying the projection layer to `pooler_output`.
            Returned when `with_projection=True` is set during model initialization.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, output_dim)`
            - **Description**: Text embeddings representing the text features.
        
        last_hidden_state (torch.FloatTensor, required, shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden states at the output of the last layer of the model.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, sequence_length, hidden_size)`
            - **Description**: Contains the hidden states from the last layer, used for subsequent tasks.
        
        hidden_states (tuple(torch.FloatTensor), optional):
            Returned when `output_hidden_states=True` is passed or `config.output_hidden_states=True`.
            - **Type**: `tuple(torch.FloatTensor)`
            - **Description**: Tuple of hidden states for each layer, plus the initial embedding output.
            - **Shape**: `(batch_size, sequence_length, hidden_size)` for each tuple element.
        
        attentions (tuple(torch.FloatTensor), optional):
            Returned when `output_attentions=True` is passed or `config.output_attentions=True`.
            - **Type**: `tuple(torch.FloatTensor)`
            - **Description**: Tuple of attention weights after softmax, used to compute the weighted average in self-attention heads.
            - **Shape**: `(batch_size, num_heads, sequence_length, sequence_length)` for each tuple element.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Siglip2Output(BaseModelOutput):
    """
    Output of the Siglip2 model, including image-text contrastive loss, similarity scores,
    embeddings, and sub-model outputs.

    Args:
        loss (torch.FloatTensor, optional, shape `(1,)`):
            Returned when `return_loss=True`, this is the contrastive loss for image-text similarity.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(1,)`
            - **Description**: Contrastive loss, used to measure similarity between images and text.
        
        logits_per_image (torch.FloatTensor, required, shape `(image_batch_size, text_batch_size)`):
            Scaled dot-product scores between `image_embeds` and `text_embeds`, representing image-text similarity scores.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(image_batch_size, text_batch_size)`
            - **Description**: Image-text similarity scores, used to evaluate the match between images and text.
        
        logits_per_text (torch.FloatTensor, required, shape `(text_batch_size, image_batch_size)`):
            Scaled dot-product scores between `text_embeds` and `image_embeds`, representing text-image similarity scores.
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(text_batch_size, image_batch_size)`
            - **Description**: Text-image similarity scores, used to evaluate the match between text and images.
        
        text_embeds (torch.FloatTensor, required, shape `(batch_size, output_dim)`):
            Text embeddings obtained by applying the projection layer to the pooled output of [`Siglip2TextModel`].
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, output_dim)`
            - **Description**: Text embeddings representing the text features.
        
        image_embeds (torch.FloatTensor, required, shape `(batch_size, output_dim)`):
            Image embeddings obtained by applying the projection layer to the pooled output of [`Siglip2VisionModel`].
            - **Type**: `torch.FloatTensor`
            - **Shape**: `(batch_size, output_dim)`
            - **Description**: Image embeddings representing the image features.
        
        text_model_output (BaseModelOutputWithPooling):
            Output of the [`Siglip2TextModel`].
            - **Type**: `BaseModelOutputWithPooling`
            - **Description**: Contains detailed output information from the text model, such as hidden states.
        
        vision_model_output (BaseModelOutputWithPooling):
            Output of the [`Siglip2VisionModel`].
            - **Type**: `BaseModelOutputWithPooling`
            - **Description**: Contains detailed output information from the vision model, such as hidden states.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Converts the Siglip2Output object to a tuple.

        Returns:
            Tuple[Any]: A tuple containing the attribute values of the Siglip2Output object.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Siglip2VisionEmbeddings(nn.Module):
    """
    Siglip2 visual embedding module, used to convert image pixel values into embedding vectors
    and add positional embeddings.

    Args:
        config (Siglip2VisionConfig): 
            The configuration object for the vision model, containing various parameters
            like hidden size, patch size, etc.
    """
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        # Embedding dimension,
        self.embed_dim = config.hidden_size
        # Patch size (height and width of each image patch)
        self.patch_size = config.patch_size

        # Define a linear layer to map each image patch (pixel block) to an embedding vector
        self.patch_embedding = nn.Linear(
            # Input features: num_channels * patch_size^2
            in_features=config.num_channels * self.patch_size * self.patch_size,
            # Output features: embedding dimension
            out_features=self.embed_dim,
        )

        # Total number of patches the image is divided into
        self.num_patches = config.num_patches
        # Calculate the grid size for positional embeddings (assuming a square image)
        self.position_embedding_size = int(self.num_patches**0.5)
        # Define an embedding layer to store positional embeddings for each patch
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resizes positional embeddings to fit specific image dimensions and pads to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                The positional embedding tensor, shape (height, width, embed_dim).
            spatial_shapes (`torch.LongTensor`):
                Spatial shape tensor, shape (batch_size, 2), used to resize positional embeddings.
                Each element contains [target_height, target_width].
            max_length (`int`):
                The maximum length after padding, ensuring all batch elements have the same length.

        Returns:
            `torch.Tensor`: 
                The resized and padded embedding tensor, shape (batch_size, max_length, embed_dim).
        """
        # Get batch size
        batch_size = spatial_shapes.shape[0]
        # Get embedding dimension
        embed_dim = positional_embeddings.shape[-1]
        # Record original data type
        source_dtype = positional_embeddings.dtype

        # Create an empty tensor to store the resized positional embeddings
        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # Permute positional embeddings from (height, width, embed_dim) to (embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # If on CPU, upcast to float32 as CPU does not support antialias for bfloat16/float16
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # Get target height and width for the current batch item
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            # Bilinearly interpolate positional embeddings to the target size
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # Reshape resized embeddings from (1, embed_dim, height, width) to (height * width, embed_dim)
            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Convert data type back to original
            resized_embeddings = resized_embeddings.to(source_dtype)

            # Fill the resized embeddings into the result tensor
            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            # Pad the remaining part with the first positional embedding
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass to generate image embeddings.

        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values tensor, shape (batch_size, max_num_patches, num_channels * patch_size^2).
            spatial_shapes (`List[Tuple[int, int]]`):
                List of spatial shapes, shape (batch_size, 2), used to resize positional embeddings.
                Each element contains [height, width].

        Returns:
            `torch.Tensor`: 
                Generated image embedding tensor, shape (batch_size, max_num_patches, embed_dim).
        """
        # Convert pixel values tensor to the same type as patch_embedding weights
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional embeddings and reshape to (height, width, embed_dim)
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )

        # Resize positional embeddings to fit specific image dimensions and pad to fixed size
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings to get final image embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """
    Multi-head attention mechanism from the paper 'Attention Is All You Need'.

    Args:
        config: 
            Model configuration object with the following attributes:
            - hidden_size (int): Hidden size, also the embedding dimension for attention.
            - num_attention_heads (int): Number of attention heads.
            - attention_dropout (float): Dropout probability for attention weights.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding dimension for attention,
        self.embed_dim = config.hidden_size
        # Number of attention heads
        self.num_heads = config.num_attention_heads
        # Dimension of each attention head
        self.head_dim = self.embed_dim // self.num_heads

        # Check if embed_dim is divisible by num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        # Scaling factor for attention scores
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # Define linear layers for query, key, and value projections
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Output projection layer
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor, shape (batch_size, sequence_length, hidden_size).
            attention_mask (`torch.Tensor`, optional):
                Attention mask tensor to mask certain positions, shape (batch_size, 1, sequence_length, sequence_length).
            output_attentions (`bool`, optional):
                Whether to return attention weights.

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor]]`:
                - `attn_output`: Attention output, shape (batch_size, sequence_length, hidden_size).
                - `attn_weights`: Attention weights, if `output_attentions` is `True`, shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        # Get batch size and sequence length
        batch_size, q_len, _ = hidden_states.size()

        # Calculate query, key, and value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape query, key, and value tensors for multi-head attention
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get key and value sequence lengths
        k_v_seq_len = key_states.shape[-2]
        # Calculate raw attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Check if attention weights shape is correct
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # If attention mask is provided, add it to the attention scores
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Convert attention scores to float32 for softmax, then back to original dtype
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Calculate final attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Check if attention output shape is correct
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape attention output tensor for subsequent processing
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # Apply output projection layer
        attn_output = self.out_proj(attn_output)

        # Return attention output and weights (if needed)
        return attn_output, attn_weights


class Siglip2FlashAttention2(Siglip2Attention):
    """
    Flash Attention module for Siglip2Attention. This module inherits from `Siglip2Attention`,
    so the model weights remain unchanged.
    The only modification needed is in the forward pass to correctly call the Flash Attention
    public API and handle potential padding tokens in the input.

    Attributes:
        is_causal (bool): 
            Whether the attention is causal. If `True`, only allows attention to current and previous tokens.
    """

    is_causal = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This is needed to ensure compatibility with the logic in PreTrainedModel
        self._flash_attn_uses_top_left_mask = True


    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Flash Attention.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor, shape (batch_size, sequence_length, hidden_size).
            attention_mask (`torch.LongTensor`, optional):
                Attention mask tensor, shape (batch_size, 1, sequence_length, sequence_length).
            output_attentions (`bool`):
                Whether to output attention weights.

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`:
                - `attn_output`: Attention output, shape (batch_size, sequence_length, hidden_size).
                - `attn_weights`: Attention weights (None for Flash Attention).
        """
        if output_attentions:
             logger.warning_once(
                "Siglip2FlashAttention2 does not support outputting attention weights. Setting output_attentions=False."
            )
        output_attentions = False

        # Get batch size and sequence length
        batch_size, q_len, _ = hidden_states.size()

        # Calculate query, key, and value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention requires input shape (batch_size, seq_length, num_heads, head_dim)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, we often convert LayerNorm layers to float32 for stability.
        # This can silently upcast the input hidden states. We need to cast them back.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle model quantization
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

        # Call Flash Attention's forward method
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

        # Reshape attention output tensor
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        # Flash Attention does not return weights
        attn_weights = None

        # Return attention output
        return attn_output, attn_weights


class Siglip2SdpaAttention(Siglip2Attention):
    """
    Siglip2 attention module using torch.nn.functional.scaled_dot_product_attention.
    This module inherits from `Siglip2Attention` as the weights remain the same.
    The only change is in the forward pass to adapt to the SDPA API.

    Attributes:
        is_causal (bool): 
            Whether the attention is causal.
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
        Forward pass for SDPA attention.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor, shape (batch_size, sequence_length, hidden_size).
            attention_mask (`torch.Tensor`, optional):
                Attention mask tensor, shape (batch_size, 1, sequence_length, sequence_length).
            output_attentions (`bool`, optional):
                Whether to output attention weights.

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor]]`:
                - `attn_output`: Attention output, shape (batch_size, sequence_length, hidden_size).
                - `attn_weights`: Attention weights (None if `output_attentions=False`).
        """
        if output_attentions:
            # SDPA does not support returning attention weights, fallback to eager
            logger.warning_once(
                "Siglip2SdpaAttention does not support outputting attention weights. Falling back to Siglip2Attention."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        # Get batch size and sequence length
        batch_size, q_len, _ = hidden_states.size()

        # Calculate query, key, and value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape tensors for multi-head attention
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We use the `is_causal` flag instead of an inline condition in SDPA
        # to support torch.compile's dynamic shapes and fullgraph options.
        is_causal = True if self.is_causal and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape attention output tensor
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        # Apply output projection layer
        attn_output = self.out_proj(attn_output)

        # Return attention output, no weights
        return attn_output, None


class Siglip2MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) module for Siglip2, used for non-linear transformation
    after the attention mechanism.

    Args:
        config: 
            Model configuration object with the following attributes:
            - hidden_size (int): Hidden size.
            - intermediate_size (int): Dimension of the intermediate layer in the MLP.
            - hidden_act (str): Activation function type, e.g., "relu", "gelu".
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Select activation function based on config
        self.activation_fn = ACT2FN[config.hidden_act]
        # Define the first fully connected layer (hidden -> intermediate)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Define the second fully connected layer (intermediate -> hidden)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, applying the MLP transformation.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor, shape `(batch_size, seq_length, hidden_size)`.

        Returns:
            `torch.Tensor`: 
                Transformed tensor, shape `(batch_size, seq_length, hidden_size)`.
        """
        # Apply first fully connected layer
        hidden_states = self.fc1(hidden_states)
        # Apply activation function
        hidden_states = self.activation_fn(hidden_states)
        # Apply second fully connected layer
        hidden_states = self.fc2(hidden_states)
        # Return transformed tensor
        return hidden_states


# Mapping of attention implementation names to classes
SIGLIP2_ATTENTION_CLASSES = {
    "eager": Siglip2Attention,
    "flash_attention_2": Siglip2FlashAttention2,
    "sdpa": Siglip2SdpaAttention,
}


class Siglip2EncoderLayer(nn.Module):
    """
    Encoder layer for Siglip2, containing a self-attention mechanism and an MLP module.

    Args:
        config (Siglip2Config): 
            Model configuration object with the following attributes:
            - hidden_size (int): Hidden size.
            - intermediate_size (int): MLP intermediate layer dimension.
            - hidden_act (str): Activation function type.
            - layer_norm_eps (float): Epsilon value for LayerNorm.
            - _attn_implementation (str): Attention implementation, e.g., "eager", "flash_attention_2", "sdpa".
    """
    def __init__(self, config: Siglip2Config):
        super().__init__()

        # Embedding dimension,
        self.embed_dim = config.hidden_size
        # Instantiate the attention mechanism based on config
        self.self_attn = SIGLIP2_ATTENTION_CLASSES[config._attn_implementation](config=config)
        # Define the first LayerNorm
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Instantiate the MLP module
        self.mlp = Siglip2MLP(config)
        # Define the second LayerNorm
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Forward pass, processing the input tensor through self-attention and MLP.

        Args:
            hidden_states (`torch.FloatTensor`):
                Input tensor, shape `(batch_size, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask, shape `(batch_size, 1, q_len, k_v_seq_len)`, where padding is indicated by large negative values.
            output_attentions (`bool`, optional, default=`False`):
                Whether to return attention weights from all attention layers.

        Returns:
            `Tuple[torch.FloatTensor]`: 
                - `hidden_states`: Transformed hidden states, shape `(batch_size, seq_len, embed_dim)`.
                - `attn_weights`: Attention weights, if `output_attentions` is `True`, shape `(batch_size, num_heads, q_len, k_v_seq_len)`.
        """
        # Save input for residual connection
        residual = hidden_states

        # Apply first LayerNorm
        hidden_states = self.layer_norm1(hidden_states)
        # Apply self-attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # Residual connection
        hidden_states = residual + hidden_states

        # Save input for residual connection
        residual = hidden_states

        # Apply second LayerNorm
        hidden_states = self.layer_norm2(hidden_states)
        # Apply MLP
        hidden_states = self.mlp(hidden_states)
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Package outputs
        outputs = (hidden_states,)

        if output_attentions:
            # Add attention weights to output if needed
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder composed of `config.num_hidden_layers` self-attention layers.
    Each layer is an instance of [`Siglip2EncoderLayer`].

    Args:
        config (Siglip2Config): 
            Model configuration object with attributes:
            - num_hidden_layers (int): Number of layers in the encoder.
            - Other config params: hidden_size, intermediate_size, hidden_act, layer_norm_eps, output_attentions, output_hidden_states, use_return_dict, etc.
    """

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
        """
        Forward pass, processing input embeddings through multiple self-attention layers.

        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally pass embedded representations directly, instead of `input_ids`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - `1` for tokens that are **not masked**,
                - `0` for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether to return attention weights of all attention layers. See `attentions` in the returned tensors.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states of all layers. See `hidden_states` in the returned tensors.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `Union[Tuple, BaseModelOutput]`: 
                - If `return_dict=True`, returns a `BaseModelOutput` object.
                - Otherwise, returns a tuple with `hidden_states`, `encoder_states`, `all_attentions`.
        """
        # Set output flags based on config or arguments
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize containers for hidden states and attention weights
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Set initial hidden state to input embeddings
        hidden_states = inputs_embeds

        # Iterate through all encoder layers
        for encoder_layer in self.layers:
            if output_hidden_states:
                # Store current hidden state
                encoder_states = encoder_states + (hidden_states,)

            # Use gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # Otherwise, call the encoder layer's forward method normally
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # Update hidden state
            hidden_states = layer_outputs[0]

            if output_attentions:
                # Store attention weights
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            # Store final hidden state
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Docstring for vision input arguments
SIGLIP2_VISION_INPUTS_DOCSTRING = r"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Pixel values. Will be ignored if padding is provided. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
    output_attentions (`bool`, *optional*):
        Whether to return attention weights of all attention layers. See `attentions` in the returned tensors.
    output_hidden_states (`bool`, *optional*):
        Whether to return hidden states of all layers. See `hidden_states` in the returned tensors.
    interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
        Whether to interpolate pre-trained positional encodings.
    return_dict (`bool`, *optional*):
        Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Siglip2VisionTransformer(nn.Module):
    """
    Siglip2 Visual Transformer model, including image embeddings, encoder, LayerNorm,
    and an optional head module.

    Args:
        config (Siglip2VisionConfig): 
            Configuration object for the vision model, with attributes:
            - hidden_size (int): Hidden size.
            - layer_norm_eps (float): Epsilon for LayerNorm.
            - vision_use_head (bool): Whether to use the head module.
            - Other params: num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob, etc.
    """
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        # Embedding dimension
        embed_dim = config.hidden_size

        # Instantiate visual embedding module
        self.embeddings = Siglip2VisionEmbeddings(config)
        # Instantiate encoder module
        self.encoder = Siglip2Encoder(config)
        # Instantiate post-LayerNorm
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # Determine if head module should be used
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)

        # Check if Flash Attention 2 is used
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass, processing input image pixel values through the visual Transformer.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values tensor. Will be ignored if padding is provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Attention mask to avoid attention on padding token indices.
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                Spatial shape tensor for resizing positional embeddings.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether to return a `BaseModelOutputWithPooling` instead of a tuple.

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - If `return_dict=True`, returns `BaseModelOutputWithPooling`.
                - Otherwise, returns a tuple: `(last_hidden_state, pooler_output, hidden_states, attentions)`.
        """
        # Set output flags
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Apply visual embedding module
        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        # Process attention mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        # Apply encoder module
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get last hidden state
        last_hidden_state = encoder_outputs[0]
        # Apply post-LayerNorm
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Apply head module (if used)
        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        
        # Return outputs based on return_dict
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2TextEmbeddings(nn.Module):
    """
    Text embedding module for Siglip2, converting input token IDs to embedding vectors
    and adding positional embeddings.

    Args:
        config (Siglip2TextConfig): 
            Configuration object for the text model, with attributes:
            - vocab_size (int): Vocabulary size.
            - hidden_size (int): Hidden size.
            - max_position_embeddings (int): Maximum positional embedding length.
    """
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        # Embedding dimension
        embed_dim = config.hidden_size

        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # Positional embedding layer
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # Register a buffer for position IDs, not serialized when saving the model
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass, converting token IDs or embeddings into final embeddings with positions.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token ID tensor, shape `(batch_size, sequence_length)`.
            position_ids (`torch.LongTensor`, *optional*):
                Position ID tensor, shape `(batch_size, sequence_length)`. Generated automatically if not provided.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed input embeddings, shape `(batch_size, sequence_length, hidden_size)`. If provided, `input_ids` is ignored.

        Returns:
            `torch.Tensor`: 
                Embedding vectors with positional info, shape `(batch_size, sequence_length, hidden_size)`.
        """
        # Get sequence length
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # Get max position embedding length
        max_position_embedding = self.position_embedding.weight.shape[0]

        # Check if sequence length exceeds max
        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Get token embeddings if inputs_embeds is not provided
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # Get positional embeddings
        position_embeddings = self.position_embedding(position_ids)

        # Add token and positional embeddings
        embeddings = inputs_embeds + position_embeddings

        return embeddings


def _trunc_normal_(tensor, mean, std, a, b):
    """
    Truncated normal initialization for the input tensor.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        a (float): Lower bound of truncation.
        b (float): Upper bound of truncation.
    """
    def norm_cdf(x):
        """
        Compute the cumulative distribution function (CDF) of the standard normal distribution.

        Args:
            x (float): Input value.

        Returns:
            float: CDF value of the standard normal distribution.
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    # Check if mean is more than 2 std devs from the bounds
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Generate values by using truncated uniform, then inverse CDF of normal
    # Get upper and lower CDF values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Fill tensor uniformly in the range [2l-1, 2u-1]
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse error function (erfinv) for inverse CDF, getting truncated std normal
    tensor.erfinv_()

    # Transform distribution to specified mean and std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure values are within bounds
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """
    Fills the input tensor with values from a truncated normal distribution. Values are drawn from
    :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`, and values outside :math:`[a, b]` are re-drawn.
    This works best when :math:`a \\leq \\text{mean} \\leq b`.

    Note: This 'tf' variant is closer to TensorFlow/JAX implementation, where bounds [a, b]
    are applied to a standard normal (mean=0, std=1.0) sample, which is then scaled and shifted.

    Args:
        tensor (torch.Tensor): An n-dimensional `torch.Tensor`.
        mean (float): Mean of the normal distribution (default: 0.0).
        std (float): Standard deviation of the normal distribution (default: 1.0).
        a (float): Minimum truncation value (default: -2.0).
        b (float): Maximum truncation value (default: 2.0).
    """
    with torch.no_grad():
        # Truncated init with standard normal (mean=0, std=1.0)
        _trunc_normal_(tensor, 0, 1.0, a, b)
        # Scale and shift the distribution
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """
    Variance scaling initialization method.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        scale (float, optional): Scaling factor (default: 1.0).
        mode (str, optional): Scaling mode: 'fan_in', 'fan_out', or 'fan_avg' (default: 'fan_in').
        distribution (str, optional): Distribution type: 'truncated_normal', 'normal', or 'uniform' (default: 'normal').
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # Std dev constant for std normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """
    LeCun normal initialization.
    Uses variance scaling with mode 'fan_in' and 'truncated_normal' distribution.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
    """
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    """
    Default Flax embedding initialization.
    Uses variance scaling with mode 'fan_in' and 'normal' distribution.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
    """
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


class Siglip2TextTransformer(nn.Module):
    """
    Text Transformer model for Siglip2, including text embeddings, encoder,
    final LayerNorm, and a linear head.

    Args:
        config (Siglip2TextConfig): 
            Configuration object for the text model, with attributes:
            - hidden_size (int): Hidden size.
            - projection_size (int): Output dimension of the linear head.
            - layer_norm_eps (float): Epsilon for LayerNorm.
            - _attn_implementation (str): Attention implementation.
            - Other params: num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob, etc.
    """
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        self.config = config
        # Embedding dimension
        embed_dim = config.hidden_size
        # Instantiate text embedding module
        self.embeddings = Siglip2TextEmbeddings(config)
        # Instantiate encoder module
        self.encoder = Siglip2Encoder(config)
        # Instantiate final LayerNorm
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # Define linear head to map hidden states to projection size
        self.head = nn.Linear(embed_dim, config.projection_size)
        # Check if Flash Attention 2 is used
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass, processing input text through the text Transformer.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens. Must be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask to avoid attention on padding token indices.
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Position indices for each token. Generated automatically if not provided.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether to return a `BaseModelOutputWithPooling` instead of a tuple.

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - If `return_dict=True`, returns `BaseModelOutputWithPooling`.
                - Otherwise, returns a tuple: `(last_hidden_state, pooler_output, hidden_states, attentions)`.
        """
        # Set output flags
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # Get input shape
        input_shape = input_ids.size()
        # Reshape tensor
        input_ids = input_ids.view(-1, input_shape[-1])

        # Apply text embedding module
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # Note: Siglip2's text model does not use a causal mask like the original CLIP.
        # Expand attention mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # Apply encoder module
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get last hidden state
        last_hidden_state = encoder_outputs[0]
        # Apply final LayerNorm
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, the last token is always EOS.
        # Take the hidden state of the last token as pooled output
        pooled_output = last_hidden_state[:, -1, :]
        # Apply linear head
        pooled_output = self.head(pooled_output)

        if not return_dict:
            # Return tuple output
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weight initialization and provide a simple interface
    for downloading and loading pre-trained models.

    Attributes:
        config_class (Siglip2Config): Configuration class.
        base_model_prefix (str): Model prefix for saving and loading.
        supports_gradient_checkpointing (bool): Whether gradient checkpointing is supported.
    """

    config_class = Siglip2Config
    base_model_prefix = "siglip2"
    supports_gradient_checkpointing = True

    # List of modules not to split
    _no_split_modules = [
        "Siglip2TextEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2MultiheadAttentionPoolingHead",
    ]
    # Whether Flash Attention 2 is supported
    _supports_flash_attn_2 = True
    # Whether SDPA is supported
    _supports_sdpa = True

    def _init_weights(self, module):
        """
        Initialize model weights.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, Siglip2VisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Siglip2Config)
                else self.config.hidden_size
            )
            # Initialize positional embedding weights
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)  # Use default Flax embedding init
        elif isinstance(module, Siglip2Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)  # Init query projection weights
            nn.init.xavier_uniform_(module.k_proj.weight)  # Init key projection weights
            nn.init.xavier_uniform_(module.v_proj.weight)  # Init value projection weights
            nn.init.xavier_uniform_(module.out_proj.weight) # Init output projection weights
            nn.init.zeros_(module.q_proj.bias)  # Init query projection bias
            nn.init.zeros_(module.k_proj.bias)  # Init key projection bias
            nn.init.zeros_(module.v_proj.bias)  # Init value projection bias
            nn.init.zeros_(module.out_proj.bias)  # Init output projection bias
        elif isinstance(module, Siglip2MLP):
            nn.init.xavier_uniform_(module.fc1.weight)  # Init first FC layer weights
            nn.init.xavier_uniform_(module.fc2.weight)  # Init second FC layer weights
            nn.init.normal_(module.fc1.bias, std=1e-6)  # Init first FC layer bias
            nn.init.normal_(module.fc2.bias, std=1e-6)  # Init second FC layer bias
        elif isinstance(module, Siglip2MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)  # Init probe data
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)  # Init attention input projection weights
            nn.init.zeros_(module.attention.in_proj_bias.data)  # Init attention input projection bias
        elif isinstance(module, Siglip2Model):
            logit_scale_init = torch.log(torch.tensor(1.0))  # Init logit scale
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()  # Init logit bias
        elif isinstance(module, Siglip2ForImageClassification):
            # Init classifier weights
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # Use LeCun normal init for Linear or Conv layers
            lecun_normal_(module.weight)
            if module.bias is not None:
                # Init bias to zero
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Init LayerNorm bias to zero
            module.bias.data.zero_()
            # Init LayerNorm weight to 1.0
            module.weight.data.fill_(1.0)


class Siglip2TextModel(Siglip2PreTrainedModel):
    """
    Text model for Siglip2, based on `Siglip2PreTrainedModel`.

    Args:
        config (Siglip2TextConfig): 
            Configuration object for the text model, with attributes:
            - hidden_size (int): Hidden size.
            - projection_size (int): Output dimension of the linear head.
            - layer_norm_eps (float): Epsilon for LayerNorm.
            - use_return_dict (bool): Whether to use `ModelOutput` for results.
            - Other params: num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob, etc.
    """
    config_class = Siglip2TextConfig

    def __init__(self, config: Siglip2TextConfig):
        super().__init__(config)
        self.text_model = Siglip2TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer.

        Returns:
            nn.Module: The input embedding layer, typically `token_embedding`.
        """
        # Return the token embedding layer from the text embedding module
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        """
        Set the input embedding layer.

        Args:
            value (nn.Module): The input embedding layer to set.
        """
        # Set the token embedding layer in the text embedding module
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass, processing input text through the text model.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask.
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Position indices.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a `BaseModelOutputWithPooling`.

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - If `return_dict=True`, returns `BaseModelOutputWithPooling`.
                - Otherwise, returns a tuple: `(last_hidden_state, pooler_output, hidden_states, attentions)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the text Transformer model's forward method
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-head Attention Pooling Head.

    Args:
        config (Siglip2VisionConfig): 
            Configuration object for the vision model, with attributes:
            - hidden_size (int): Hidden size.
            - num_attention_heads (int): Number of attention heads.
            - layer_norm_eps (float): Epsilon for LayerNorm.
            - Other params: intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob, etc.
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()

        # Initialize probe parameter, shape (1, 1, hidden_size)
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # Instantiate MultiheadAttention layer
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        # Instantiate LayerNorm
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Instantiate MLP module
        self.mlp = Siglip2MLP(config)
        # Number of attention heads
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass, applying multi-head attention pooling.

        Args:
            hidden_state (`torch.Tensor`):
                Input hidden state, shape `(batch_size, sequence_length, hidden_size)`.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask, shape `(batch_size, sequence_length)`, to mask positions.

        Returns:
            `torch.Tensor`: 
                Pooled hidden state, shape `(batch_size, hidden_size)`.
        """
        # Get batch size
        batch_size = hidden_state.shape[0]
        # Repeat probe parameter to match batch size
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            # Get target and source lengths
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            # Prepare 4D attention mask
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            # Repeat mask to match multi-head
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            # Reshape mask
            attention_mask = attention_mask.reshape(-1, target_len, source_len)
        
        # Apply MultiheadAttention layer
        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        # Save input for residual connection
        residual = hidden_state
        # Apply LayerNorm
        hidden_state = self.layernorm(hidden_state)
        # Apply MLP and residual connection
        hidden_state = residual + self.mlp(hidden_state)

        # Return pooled hidden state (the first token)
        return hidden_state[:, 0]


class Siglip2VisionModel(Siglip2PreTrainedModel):
    """
    Vision model for Siglip2, based on `Siglip2PreTrainedModel`.

    Args:
        config (Siglip2VisionConfig): 
            Configuration object for the vision model, with attributes:
            - hidden_size (int): Hidden size.
            - num_attention_heads (int): Number of attention heads.
            - intermediate_size (int): MLP intermediate dimension.
            - hidden_act (str): Activation function type.
            - layer_norm_eps (float): Epsilon for LayerNorm.
            - Other params: num_hidden_layers, attention_dropout, hidden_dropout_prob, etc.
    """
    config_class = Siglip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

        # Instantiate the visual Transformer model
        self.vision_model = Siglip2VisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer.

        Returns:
            nn.Module: The input embedding layer, typically `patch_embedding`.
        """
        # Return the patch embedding layer from the visual embedding module
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass, processing input pixel values through the vision model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values tensor.
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                Spatial shape tensor for resizing positional embeddings.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a `BaseModelOutputWithPooling`.

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - If `return_dict=True`, returns `BaseModelOutputWithPooling`.
                - Otherwise, returns a tuple: `(last_hidden_state, pooler_output, hidden_states, attentions)`.
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the visual Transformer model's forward method
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Siglip2Model(Siglip2PreTrainedModel):
    """
    Siglip2 model, combining text and vision models, based on `Siglip2PreTrainedModel`.

    Args:
        config (Siglip2Config): 
            Model configuration object, with attributes:
            - text_config (Siglip2TextConfig): Text model configuration.
            - vision_config (Siglip2VisionConfig): Vision model configuration.
            - Other params: logit_scale, logit_bias, etc.
    """
    config_class = Siglip2Config

    def __init__(self, config: Siglip2Config):
        super().__init__(config)

        # Check text_config type
        if not isinstance(config.text_config, Siglip2TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type Siglip2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # Check vision_config type
        if not isinstance(config.vision_config, Siglip2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type Siglip2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # Get text config
        text_config = config.text_config
        # Get vision config
        vision_config = config.vision_config

        # First, initialize text and vision models with the correct attention implementation
        text_model = Siglip2TextModel._from_config(text_config)
        vision_model = Siglip2VisionModel._from_config(vision_config)

        # Second, get text and vision submodules (for backward compatibility)
        # Get text model's submodule
        self.text_model = text_model.text_model
        # Get vision model's submodule
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Get text features.

        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
                Text embeddings from the pooled output of [`Siglip2TextModel`].

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

        >>> # Important: Ensure padding="max_length" is set, as this is how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```
        """
        # Use config from Siglip2Model if specified
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call text model's forward method
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get pooled output
        pooled_output = text_outputs[1]

        # Return text features
        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Get image features.

        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
                Image embeddings from the pooled output of [`Siglip2VisionModel`].

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> url = "[http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```
        """
        # Use config from Siglip2Model if specified
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call vision model's forward method
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get pooled output
        pooled_output = vision_outputs[1]

        # Return image features
        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Siglip2Output]:
        """
        Forward pass, processing input text and images through the Siglip2 model.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
        >>> url = "[http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # Important: We pass `padding=max_length` as the model was trained this way
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # These are probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```
        """
        # Use config from Siglip2Model if specified
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call vision model's forward method
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Call text model's forward method
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get image embeddings
        image_embeds = vision_outputs[1]
        # Get text embeddings
        text_embeds = text_outputs[1]

        # Normalize features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Calculate cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        # Transpose to get image-text logits
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Calculate Sigmoid Loss
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        # Return Siglip2Output object
        return Siglip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class Siglip2ForImageClassification(Siglip2PreTrainedModel):
    """
    Image classification model for Siglip2, based on `Siglip2PreTrainedModel`.

    Args:
        config (Siglip2Config): 
            Model configuration object, with attributes:
            - vision_config (Siglip2VisionConfig): Vision model configuration.
            - num_labels (int): Number of classification labels.
            - Other params: problem_type, etc.
    """
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2Config) -> None:
        super().__init__(config)

        # Number of classification labels
        self.num_labels = config.num_labels

        # Create vision model with correct attention implementation and get submodule
        vision_model = Siglip2VisionModel._from_config(config.vision_config)
        self.vision_model = vision_model.vision_model

        # Classifier head
        self.classifier = (
            # Use Linear layer if num_labels > 0, otherwise Identity
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass, processing input images through the image classification model.

        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
                Input pixel values tensor.
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask.
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
                Spatial shape tensor.
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for calculating loss. Indices should be in `[0, ..., config.num_labels - 1]`.
                If `config.num_labels == 1`, regression loss (MSE) is computed.
                If `config.num_labels > 1`, classification loss (CrossEntropy) is computed.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return an `ImageClassifierOutput`.

        Returns:
            `Union[Tuple, ImageClassifierOutput]`: 
                - If `return_dict=True`, returns `ImageClassifierOutput`.
                - Otherwise, returns a tuple: `(logits, hidden_states, attentions)`.
        """
        # Set output flags
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call vision model's forward method
        outputs = self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get sequence output
        sequence_output = outputs[0]

        # Average pooling of patch tokens
        if pixel_attention_mask is not None:
            # Prepare pooling mask
            pool_mask = pixel_attention_mask[..., None].to(sequence_output.device)
            # Apply masked pooling
            sequence_output = torch.sum(sequence_output * pool_mask, dim=1) / torch.sum(pool_mask, dim=1)
        else:
            # Otherwise, direct average pooling
            sequence_output = torch.mean(sequence_output, dim=1)

        # Apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Move labels to correct device
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # Set problem type to regression
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # Set problem type to single-label classification
                    self.config.problem_type = "single_label_classification"
                else:
                    # Set problem type to multi-label classification
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # Use MSE loss
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # Calculate regression loss
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # Use CrossEntropy loss
                loss_fct = CrossEntropyLoss()
                # Calculate classification loss
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # Use BCEWithLogits loss
                loss_fct = BCEWithLogitsLoss()
                # Calculate multi-label classification loss
                loss = loss_fct(logits, labels)

        if not return_dict:
            # Return tuple output
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return encapsulated model output
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
