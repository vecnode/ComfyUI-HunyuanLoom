import torch
from torch import Tensor
from einops import repeat


class vecnode_TweakHunyuan:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "temporal_rope_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "temporal_rope_offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "temporal_attn_entropy_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "expose_internals": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "hunyuanloom"

    def patch(self, model, temporal_rope_factor, temporal_rope_offset, temporal_attn_entropy_scale, expose_internals):
        m = model.clone()
        
        # Get the diffusion model (HunyuanVideo transformer)
        diffusion_model = m.model.diffusion_model
        
        # Store original forward method
        original_forward = diffusion_model.forward
        
        # Check if model has forward_orig and patch it directly
        if hasattr(diffusion_model, 'forward_orig'):
            # Store original forward_orig
            original_forward_orig = diffusion_model.forward_orig
            
            # Create wrapper to modify img_ids in forward_orig
            # Accept **kwargs to handle additional parameters that may be passed by different model versions
            def tweaked_forward_orig(img, img_ids, txt, txt_ids, txt_mask, timesteps, y, *args, **kwargs):
                # Extract parameters from kwargs that might be passed as keywords
                # But don't extract if they're already in *args (to avoid duplicates)
                # The actual call pattern is: forward_orig(x, img_ids, context, txt_ids, attention_mask, timestep, y, txt_byt5, guidance, guiding_frame_index, ref_latent, disable_time_r=..., control=..., transformer_options=...)
                # So args contains: txt_byt5, guidance, guiding_frame_index, ref_latent
                # And kwargs contains: disable_time_r, control, transformer_options
                
                # Extract only from kwargs (not from args) to avoid duplicates
                control = kwargs.pop('control', None)
                transformer_options = kwargs.pop('transformer_options', {})
                
                # Get parameters from transformer_options (set below) or use closure values
                rope_factor = transformer_options.get('temporal_rope_factor', temporal_rope_factor)
                rope_offset = transformer_options.get('temporal_rope_offset', temporal_rope_offset)
                entropy_scale = transformer_options.get('temporal_attn_entropy_scale', temporal_attn_entropy_scale)
                expose = transformer_options.get('expose_internals', expose_internals)
                
                # Modify temporal coordinates in img_ids for RoPE warping
                # img_ids shape: [batch, seq_len, 3] where last dim is [time, height, width]
                if rope_factor != 1.0 or rope_offset != 0.0:
                    img_ids_modified = img_ids.clone()
                    # Modify time coordinate (first channel of last dimension)
                    img_ids_modified[:, :, 0] = img_ids_modified[:, :, 0] * rope_factor + rope_offset
                    img_ids = img_ids_modified
                
                # Store internals if requested
                if expose:
                    transformer_options['hunyuan_internals'] = {
                        'img_ids': img_ids,
                        'img': img,
                        'txt': txt,
                    }
                
                # Add temporal attention entropy tweaking to transformer_options
                if entropy_scale != 1.0:
                    transformer_options['temporal_attn_entropy_scale'] = entropy_scale
                
                # Call original forward_orig with modified img_ids
                # Pass *args as-is (contains txt_byt5, guidance, guiding_frame_index, ref_latent)
                # Only pass control and transformer_options as keywords if they weren't in args
                # Pass all remaining kwargs through
                return original_forward_orig(img, img_ids, txt, txt_ids, txt_mask, timesteps, y, *args, control=control, transformer_options=transformer_options, **kwargs)
            
            # Patch forward_orig directly
            diffusion_model.forward_orig = tweaked_forward_orig
        else:
            # For base HunyuanVideo without forward_orig, patch forward method
            def tweaked_forward(x, timestep, context, y, guidance, attention_mask=None, control=None, transformer_options={}, **kwargs):
                # Get parameters from transformer_options (set below) or use closure values
                rope_factor = transformer_options.get('temporal_rope_factor', temporal_rope_factor)
                rope_offset = transformer_options.get('temporal_rope_offset', temporal_rope_offset)
                entropy_scale = transformer_options.get('temporal_attn_entropy_scale', temporal_attn_entropy_scale)
                expose = transformer_options.get('expose_internals', expose_internals)
                
                # Intercept img_ids creation (same logic as ModifiedHunyuanVideo.forward)
                bs, c, t, h, w = x.shape
                patch_size = diffusion_model.patch_size
                t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
                h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
                w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
                img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
                img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
                img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
                img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
                
                # Modify temporal coordinates in img_ids for RoPE warping
                if rope_factor != 1.0 or rope_offset != 0.0:
                    img_ids_modified = img_ids.clone()
                    # Modify time coordinate (first channel of last dimension)
                    img_ids_modified[:, :, 0] = img_ids_modified[:, :, 0] * rope_factor + rope_offset
                    img_ids = img_ids_modified
                
                # Store internals if requested
                if expose:
                    transformer_options['hunyuan_internals'] = {
                        'img_ids': img_ids,
                        'x': x,
                        'context': context,
                    }
                
                # Add temporal attention entropy tweaking to transformer_options
                if entropy_scale != 1.0:
                    transformer_options['temporal_attn_entropy_scale'] = entropy_scale
                
                # Pass through with transformer_options
                transformer_options['temporal_rope_factor'] = rope_factor
                transformer_options['temporal_rope_offset'] = rope_offset
                transformer_options['temporal_attn_entropy_scale'] = entropy_scale
                transformer_options['expose_internals'] = expose
                return original_forward(x, timestep, context, y, guidance, attention_mask, control, transformer_options, **kwargs)
            
            # Patch the forward method
            diffusion_model.forward = tweaked_forward
        
        # Store parameters in model_options for access during forward pass
        model_options = m.model_options.copy()
        transformer_options = model_options.get('transformer_options', {}).copy()
        
        transformer_options['temporal_rope_factor'] = temporal_rope_factor
        transformer_options['temporal_rope_offset'] = temporal_rope_offset
        transformer_options['temporal_attn_entropy_scale'] = temporal_attn_entropy_scale
        transformer_options['expose_internals'] = expose_internals
        
        model_options['transformer_options'] = transformer_options
        m.model_options = model_options
        
        return (m,)

