# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

# Rewrite to propagate the position ids to all Rotary Embedding nodes within Phi4 mini reasoning correctly.

class Phi4CosSinCacheFusion(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(remove_nodes=False)
        


    def pattern(self, op, x, position_ids, attention_mask, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype):
        """
        Pattern to match Phi4's LongRoPE computation
        """
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
            x,
            cos_4d,
            sin_4d,
            _domain="com.microsoft"
        )
    
    def check(self, context, x, position_ids, attention_mask, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype, **_) -> pattern.MatchResult:
            check_result = pattern.MatchResult()                
            return check_result

    def rewrite(self, op, x, position_ids, inv_freq_base, cos_cache, sin_cache, extra_dims, dtype, **_):
        """
        Simplified rewrite for testing
        """
        print("Pattern matched! Rewriting Phi4 LongRoPE...")

        print("=" * 50)
        
        # Explore x and its producer
        print(f"x type: {type(x)}")
        print(f"x name: {getattr(x, 'name', 'No name')}")
        print(f"x shape: {getattr(x, 'shape', 'No shape')}")
        print(f"x dtype: {getattr(x, 'dtype', 'No dtype')}")
        
        if hasattr(x, 'producer') and x.producer is not None:
            producer = x.producer()
            print(f"\nProducer node:")
            print(f"  Op type: {producer.op_type}")
            print(f"  Node name: {getattr(producer, 'name', 'No name')}")
            print(f"  Num inputs: {len(producer.inputs) if hasattr(producer, 'inputs') else 'No inputs'}")
            print(f"  Num outputs: {len(producer.outputs) if hasattr(producer, 'outputs') else 'No outputs'}")
            
            # Print attributes if any
            if hasattr(producer, 'attributes'):
                print(f"  Attributes: {producer.attributes}")
            
        print("=" * 50)

        return op.RotaryEmbedding(
            x,
            position_ids,  # Simplified for testing
            cos_cache,  # Simplified for testing  
            sin_cache,  # Simplified for testing
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
