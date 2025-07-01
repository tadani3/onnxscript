# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

class Phi4RotaryEmbeddingFusion(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(name="RotaryEmbedding", as_function=True)

    def pattern(self, op, x, cos, sin, shape):
        reshaped_input = op.Reshape(x, shape, allowzero=1)
        transposed_input = op.Transpose(reshaped_input, perm=[0, 2, 1, 3])
        rotary_embedding = op.RotaryEmbedding(
            transposed_input,
            cos,
            sin,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _domain="com.microsoft",
        )
        print("Phi4RotaryEmbeddingFusion Pattern Found")
        return rotary_embedding

    def check(self, op, x, cos, sin, shape, **_) -> pattern.MatchResult:
        check_result = pattern.MatchResult()
        
        # Check input x
        if x is None or x.shape is None:
            return check_result.fail("Input x is None or has no shape information.", x)
        
        # Check cos and sin - only pass one object to fail()
        if cos is None or sin is None:
            return check_result.fail("cos or sin is None.")
        
        # Check that cos and sin have the same shape
        if cos.shape != sin.shape:
            return check_result.fail("cos and sin must have the same shape.")
        
        # Check shape input - it should be a constant tensor with 4 elements for 4D reshape
        if shape is None:
            return check_result.fail("Shape input is None.", shape)
        
        return check_result

    def rewrite(self, op, x, cos, sin, shape, **_):
        return op.RotaryEmbedding(
            x,
            cos,
            sin,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _domain="com.microsoft",
        )

_rule = Phi4RotaryEmbeddingFusion.rule()
phi4_rotary_embedding_rules = pattern.RewriteRuleSet([_rule])
fuse_phi4_rotary_embedding = _fusion_utils.apply_fusion_rules(phi4_rotary_embedding_rules)