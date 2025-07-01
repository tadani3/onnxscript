# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

# Rewrite the computation of cos/sin cache for models using a Phi3 architecture such as Phi4-mini-reasoning.

class Phi4CosSinCacheFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name: str,
        *,
        cast: bool = False,
        reshape: bool = False,
        const_freqs: bool = False,
        use_multi_cache: bool = True,
    ):
        super().__init__(name, remove_nodes=False)
        self._max_pos_id = 131072 # TODO: Need to determine how to set this value generally based on what is passed into the config file. Phi4-mini will either use 4096 or 131072
        # map from inv_freq to (cos, sin) values for transformed graph
        self._original_max_pos_id = 4096
        self._inv_freq_cos_sin_cache: dict[ir.Value, tuple[ir.Value, ir.Value]] = {}
        self._reshape = reshape
        self._cast = cast
        self._const_freqs = const_freqs
        self._use_multi_cache = use_multi_cache

    @property
    def max_pos_id(self) -> int | None:
        return self._max_pos_id

    
    @max_pos_id.setter
    def max_pos_id(self, max_pos_id: int):
        self._max_pos_id = max_pos_id  # type: ignore[assignment]

    @property 
    def original_max_pos_id(self) -> int:
        return self._original_max_pos_id
        
    @original_max_pos_id.setter
    def original_max_pos_id(self, original_max_pos_id: int):
        self._original_max_pos_id = original_max_pos_id
    
    def _compute_longrope_frequencies(self, op, config_params, sequence_length=None):
        """
        Compute the inverse frequencies with LongRoPE scaling for Phi4.
        
        Args:
            op: ONNX operation builder
            config_params: Dictionary containing model configuration parameters
            sequence_length: Optional sequence length for dynamic computation
            
        Returns:
            Tuple of (inv_freq, attention_factor)
        """
        # Extract configuration parameters
        base = config_params.get('rope_theta', 10000.0)
        partial_rotary_factor = config_params.get('partial_rotary_factor', 1.0)
        head_dim = config_params.get('head_dim', 128)
        long_factor = config_params.get('long_factor', [1.0])
        short_factor = config_params.get('short_factor', [1.0])

        print(f"Using LongRoPE with base={base}, head_dim={head_dim}, partial_rotary_factor={partial_rotary_factor}, "
              f"long_factor={long_factor}, short_factor={short_factor}, sequence_length={sequence_length}")

        # Compute dimensions
        dim = int(head_dim * partial_rotary_factor)
       
        # Compute factors for attention scaling
        original_max_position_embeddings = config_params.get('original_max_position_embeddings', self._original_max_pos_id)
        max_position_embeddings = config_params.get('max_position_embeddings', self._max_pos_id)
        
        factor = config_params.get('factor')
        if factor is None:
            factor = max_position_embeddings / original_max_position_embeddings

        attention_factor = config_params.get('attention_factor')
        if attention_factor is None and factor <= 1.0:
            attention_factor = 1.0
        
        elif attention_factor is None and factor > 1.0:
           attention_factor = np.sqrt(1 + np.log(factor) / np.log(original_max_position_embeddings))
        
        # Determine which scaling factors to use based on sequence length
        if sequence_length is not None and sequence_length > original_max_position_embeddings:
            ext_factors = np.array(long_factor, dtype=np.float32)
        else:
            ext_factors = np.array(short_factor, dtype=np.float32)
        
        frequency_range = np.arange(0, dim, 2, dtype=np.float32) / dim
        inv_freq = 1.0 / (ext_factors * (base ** frequency_range))
       
        return inv_freq, attention_factor

    def _compute_const_freqs(self, op, inv_freq, attention_factor, max_seq_len):
        """Compute cos/sin values when frequencies are constant."""
        # Generate position IDs range
        pos_ids = np.arange(max_seq_len, dtype=np.float32).reshape(-1, 1)
        inv_freq_reshaped = inv_freq.reshape(1, -1)
        
        # Compute angles
        angles = np.matmul(pos_ids, inv_freq_reshaped)
        
        # Compute cos/sin values with attention scaling
        cos_value = np.cos(angles) * attention_factor
        sin_value = np.sin(angles) * attention_factor
        
        cos_2d = op.Constant(value=ir.tensor(cos_value))
        sin_2d = op.Constant(value=ir.tensor(sin_value))
        
        return cos_2d, sin_2d
    
    def _create_multi_cache_subgraph(self, op, config_params):
        """Create the multi-cache subgraph similar to builder.py implementation."""
        # Compute short cache (for seq_len <= original_max_pos_id)
        short_inv_freq, short_attention_factor = self._compute_longrope_frequencies(
            op, config_params, seq_len=self._original_max_pos_id
        )
        cos_small, sin_small = self._compute_const_freqs(
            op, short_inv_freq, short_attention_factor, self._original_max_pos_id
        )
        
        # Compute long cache (for seq_len > original_max_pos_id)
        long_inv_freq, long_attention_factor = self._compute_longrope_frequencies(
            op, config_params, seq_len=self._max_pos_id
        )
        cos_large, sin_large = self._compute_const_freqs(
            op, long_inv_freq, long_attention_factor, self._max_pos_id
        )
        
        return cos_small, sin_small, cos_large, sin_large












    def pattern(self, op, x, position_ids, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype):
        """
        Pattern to match Phi4's LongRoPE computation
        """
        reshaped_x = op.Reshape(x, pattern.ANY_VALUE)
        transposed_x = op.Transpose(reshaped_x, perm=[0, 2, 1, 3])

        position_ids_expanded = op.Unsqueeze(position_ids, extra_dims)
        position_ids_expanded = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
        
        # Use the concatenated inv_freq
        inv_freq = pattern.OrValue([
            op.Expand(inv_freq_base, pattern.ANY_VALUE, _outputs=["expanded_inv_freq"]),
            inv_freq_base,
        ])
        
        freqs = op.MatMul(inv_freq, position_ids_expanded)
        freqs = op.Transpose(freqs, perm=[0, 2, 1])
        emb = op.Concat(freqs, freqs, axis=-1)

        cos = op.Cos(emb)
        if self._cast:
            cos = op.Cast(cos, to=dtype)

        sin = op.Sin(emb)
        if self._cast:
            sin = op.Cast(sin, to=dtype)

        cos_4d = op.Unsqueeze(cos, 1)
        sin_4d = op.Unsqueeze(sin, 1)
        
        print(f"Pattern matched: Phi4 LongRoPE with cos/sin cache computation")
        return op.RotaryEmbedding(
            transposed_x,
            cos_4d,
            sin_4d,
            _domain="com.microsoft"
        )
    
    def check(self, context, x, position_ids, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype, **_) -> pattern.MatchResult:
            check_result = pattern.MatchResult()                
            return check_result

    def rewrite(self, op, x, position_ids, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype, **_):
        """
        Simplified rewrite for testing
        """
        num_heads = x.shape[1]

        return op.RotaryEmbedding(
            x,
            position_ids,  # Simplified for testing
            cos_cache,  # Simplified for testing  
            sin_cache,  # Simplified for testing
            num_heads=num_heads,
            interleaved=0,
            _domain="com.microsoft"
        )
"""
    def check(self, x, position_ids, long_factor, short_factor, 
              base_theta, **_) -> pattern.MatchResult:
        check_result = pattern.MatchResult()
        
        # Check basic input requirements
        if not _ir_utils.has_rank(x, 3):
            return check_result.fail("Input x must be 3D.", x)
            
        if not (_ir_utils.has_rank(position_ids, 1) or _ir_utils.has_rank(position_ids, 2)):
            return check_result.fail("position_ids must be 1D or 2D.", position_ids)
            
        # Check that we have the required LongRoPE parameters
        if long_factor.const_value is None or short_factor.const_value is None:
            return check_result.fail("long_factor and short_factor must be constants.")
            
        if base_theta.const_value is None:
            return check_result.fail("base_theta must be a constant.", base_theta)
            
        return check_result

    def rewrite(self, op, x, input_ids, position_ids, attention_mask, inv_freq_base, long_factor, short_factor,
                base_theta, head_dim, partial_rotary_factor, attention_factor, **_):
        Rewrite the dynamic LongRoPE computation to use precomputed caches.
        config_params = {
            'rope_theta': base_theta.const_value.numpy().item(),
            'head_dim': head_dim.const_value.numpy().item(),
            'partial_rotary_factor': partial_rotary_factor.const_value.numpy().item(),
            'long_factor': long_factor.const_value.numpy().tolist(),
            'short_factor': short_factor.const_value.numpy().tolist(),
        }
        
        cos_small, sin_small, cos_large, sin_large = self._create_multi_cache_subgraph(op, config_params)
        
        # Create the condition
        attention_mask_shape = op.Shape(attention_mask)
        seq_len_scalar = op.Gather(attention_mask_shape, op.Constant(value_int=1))
        use_large_cache = op.Greater(seq_len_scalar, op.Constant(value_int=self._original_max_pos_id))
        
        # Use Where operations instead of If for simpler implementation
        cos_2d = op.Where(use_large_cache, cos_large, cos_small)
        sin_2d = op.Where(use_large_cache, sin_large, sin_small)

        if self._cast:
            cos_2d = op.Cast(cos_2d, to=ir.DataType.FLOAT16)  # or appropriate target dtype
            sin_2d = op.Cast(sin_2d, to=ir.DataType.FLOAT16)

        # Position IDs propagation into RotaryEmbedding nodes in Phi4-mini-reasoning graph
        minus_one_tensor = op.Constant(value_ints=[-1])  # int64[1] tensor with [-1]
        
        # Get the second dimension from the input_ids (x) shape
        input_ids_shape = op.Shape(input_ids)
        gathered_dim = op.Gather(input_ids_shape, indices=1)
        
        # Unsqueeze the gathered dimension to make it a [1] shaped tensor
        second_dim_tensor = op.Unsqueeze(gathered_dim, op.Constant(value_int=0))
        
        # Concat the two tensors to form the reshape shape
        pos_ids_reshape_tensor = op.Concat(minus_one_tensor, second_dim_tensor, axis=0)
        reshaped_position_ids = op.Reshape(position_ids, pos_ids_reshape_tensor)

        return op.RotaryEmbedding(
            x,
            reshaped_position_ids,
            cos_2d,
            sin_2d,
            _domain="com.microsoft"
        )

    def cleanup(self):
        self._inv_freq_cos_sin_cache.clear()
    """
    
# Create rule variants
_phi4_longrope_multi_cache = Phi4CosSinCacheFusion.rule("Phi4LongRoPE_multi_cache", cast=False, use_multi_cache=True)
_phi4_longrope_cast_multi_cache = Phi4CosSinCacheFusion.rule("Phi4LongRoPE_cast_multi_cache", cast=True, use_multi_cache=True)

phi4_cos_sin_cache_rules = pattern.RewriteRuleSet([
    _phi4_longrope_multi_cache,
    _phi4_longrope_cast_multi_cache
])

fuse_phi4_cos_sin_cache = _fusion_utils.apply_fusion_rules(phi4_cos_sin_cache_rules)
