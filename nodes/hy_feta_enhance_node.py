# Default attention layer indices where FETA enhancement is applied.
# 'double' and 'single' specify which transformer blocks receive FETA enhancement.
DEFAULT_ATTN = {
    'double': [i for i in range(0, 100, 1)],#[0,1,2,3,4,5,6,7,9,11,13,15,17,19,21,23,25],
    'single': [i for i in range(0, 100, 1)]
}

# FETA (Frame-to-Frame Attention) Enhancement node that improves temporal consistency in video generation.
# Analyzes cross-frame attention patterns and scales attention outputs to enhance frame-to-frame coherence.
# feta_weight controls enhancement strength: positive values increase temporal consistency,
# negative values can reduce it. Applied to specified transformer layers via attn_override.
class HYFetaEnhanceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "feta_weight": ("FLOAT", {"default": 2, "min": -100.0, "max": 100.0, "step":0.01}),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",)
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "hunyuanloom"
    FUNCTION = "apply"

    # Configures the model to apply FETA enhancement during attention computation.
    # Sets feta_weight and feta_layers in transformer_options, which are used by
    # ModifiedDoubleStreamBlock and ModifiedSingleStreamBlock to compute and apply enhancement scores.
    def apply(self, model, feta_weight, attn_override=DEFAULT_ATTN):
        model = model.clone()

        model_options = model.model_options.copy()
        transformer_options = model_options['transformer_options'].copy()

        transformer_options['feta_weight'] = feta_weight
        transformer_options['feta_layers'] = attn_override
        model_options['transformer_options'] = transformer_options

        model.model_options = model_options
        return (model,)

