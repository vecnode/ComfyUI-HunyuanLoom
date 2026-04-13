import torch
from einops import repeat


class vecnode_HunyuanSpatialWarpModel:
    """
    Patch a Hunyuan MODEL to apply spatial warping by modifying positional
    encodings (img_ids). This preserves video structure while warping space.
    
    Unlike directly warping latents, this modifies where the model "thinks"
    each spatial position is, resulting in proper video warping.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (
                    [
                        "swap_halves_vertical",
                        "swap_halves_horizontal",
                        "swap_quadrants",
                        "mirror_x",
                        "mirror_y",
                        "saliency_preserve",
                        "saliency_mirror",
                        "saliency_swap",
                        "attention_cluster_swap",
                        "attention_mirror",
                    ],
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "hunyuanloom/experimental"

    def patch(self, model, mode, strength):
        # Clone so original MODEL can still be used elsewhere
        m = model.clone()

        print("[HunyuanSpatialWarp] Starting patch...")
        print(f"[HunyuanSpatialWarp] Mode: {mode}, Strength: {strength}")

        # Get the diffusion model (HunyuanVideo transformer)
        diffusion_model = m.model.diffusion_model

        if not hasattr(diffusion_model, "forward") and not hasattr(diffusion_model, "forward_orig"):
            print("[HunyuanSpatialWarp] ERROR: diffusion_model has no forward method")
            return (m,)

        # Avoid double-patching
        if getattr(diffusion_model, "_vecnode_spatial_warp_patched", False):
            print("[HunyuanSpatialWarp] Note: model already patched; skipping re-patch.")
            return (m,)

        def warp_spatial_coords(img_ids, h_len, w_len, warp_mode, warp_strength, latent_tensor=None):
            """
            Warp spatial coordinates in img_ids.
            img_ids shape: [batch, seq_len, 3] where last dim is [time, height, width]
            We need to reshape to [t_len, h_len, w_len, 3] to apply spatial warping,
            then reshape back.
            latent_tensor: Optional latent tensor for variance-based or attention-based warping.
            """
            batch, seq_len, _ = img_ids.shape
            
            # Reshape to spatial grid: [batch, t_len, h_len, w_len, 3]
            # We need to infer t_len from seq_len
            t_len = seq_len // (h_len * w_len)
            if t_len * h_len * w_len != seq_len:
                print(f"[HunyuanSpatialWarp] WARNING: seq_len {seq_len} not divisible by h_len*w_len {h_len*w_len}")
                return img_ids
            
            img_ids_grid = img_ids.reshape(batch, t_len, h_len, w_len, 3)
            
            # Extract spatial coordinates (height and width, indices 1 and 2)
            h_coords = img_ids_grid[:, :, :, :, 1].clone()  # [batch, t_len, h_len, w_len]
            w_coords = img_ids_grid[:, :, :, :, 2].clone()  # [batch, t_len, h_len, w_len]
            
            # Apply warping to coordinates
            h_coords_warped = h_coords.clone()
            w_coords_warped = w_coords.clone()

            # here is how we will define the warping. 
            
            if warp_mode == "swap_halves_vertical":
                h_mid = h_len // 2
                # Top half (h < h_mid) points to bottom half coordinates (h + h_mid)
                # Bottom half (h >= h_mid) points to top half coordinates (h - h_mid)
                h_coords_warped[:, :, :h_mid, :] = h_coords[:, :, :h_mid, :] + h_mid
                h_coords_warped[:, :, h_mid:, :] = h_coords[:, :, h_mid:, :] - h_mid
                
            elif warp_mode == "swap_halves_horizontal":
                w_mid = w_len // 2
                # Left half (w < w_mid) points to right half coordinates (w + w_mid)
                # Right half (w >= w_mid) points to left half coordinates (w - w_mid)
                w_coords_warped[:, :, :, :w_mid] = w_coords[:, :, :, :w_mid] + w_mid
                w_coords_warped[:, :, :, w_mid:] = w_coords[:, :, :, w_mid:] - w_mid
                
            elif warp_mode == "swap_quadrants":
                h_mid = h_len // 2
                w_mid = w_len // 2
                # Rotate quadrants: TL<-BR, TR<-BL, BL<-TR, BR<-TL
                # Top-left gets bottom-right coordinates
                h_coords_warped[:, :, :h_mid, :w_mid] = h_coords[:, :, :h_mid, :w_mid] + h_mid
                w_coords_warped[:, :, :h_mid, :w_mid] = w_coords[:, :, :h_mid, :w_mid] + w_mid
                # Top-right gets bottom-left coordinates
                h_coords_warped[:, :, :h_mid, w_mid:] = h_coords[:, :, :h_mid, w_mid:] + h_mid
                w_coords_warped[:, :, :h_mid, w_mid:] = w_coords[:, :, :h_mid, w_mid:] - w_mid
                # Bottom-left gets top-right coordinates
                h_coords_warped[:, :, h_mid:, :w_mid] = h_coords[:, :, h_mid:, :w_mid] - h_mid
                w_coords_warped[:, :, h_mid:, :w_mid] = w_coords[:, :, h_mid:, :w_mid] + w_mid
                # Bottom-right gets top-left coordinates
                h_coords_warped[:, :, h_mid:, w_mid:] = h_coords[:, :, h_mid:, w_mid:] - h_mid
                w_coords_warped[:, :, h_mid:, w_mid:] = w_coords[:, :, h_mid:, w_mid:] - w_mid
                
            elif warp_mode == "mirror_x":
                # Mirror width coordinates
                w_coords_warped = w_len - 1 - w_coords
                
            elif warp_mode == "mirror_y":
                # Mirror height coordinates
                h_coords_warped = h_len - 1 - h_coords
            
            # Variance-based saliency warping modes
            elif warp_mode in ["saliency_preserve", "saliency_mirror", "saliency_swap"]:
                if latent_tensor is None:
                    print(f"[HunyuanSpatialWarp] WARNING: {warp_mode} requires latent tensor, falling back to no warp")
                else:
                    # Compute variance per spatial patch
                    # latent_tensor shape: [batch, channels, t, h, w] or [batch, seq_len, channels]
                    # We need to reshape to spatial grid
                    if latent_tensor.dim() == 5:
                        # [batch, channels, t, h, w] -> compute variance across channels
                        bs, c, t, h, w = latent_tensor.shape
                        # Downsample to match h_len, w_len if needed
                        if h != h_len or w != w_len:
                            latent_tensor = torch.nn.functional.interpolate(
                                latent_tensor.reshape(bs * t, c, h, w),
                                size=(h_len, w_len),
                                mode='bilinear',
                                align_corners=False
                            ).reshape(bs, t, c, h_len, w_len)
                            latent_tensor = latent_tensor.permute(0, 2, 1, 3, 4)  # [bs, c, t, h_len, w_len]
                        # Compute variance across channels: [batch, t, h_len, w_len]
                        patch_variance = latent_tensor.var(dim=1)  # Variance across channels
                    elif latent_tensor.dim() == 3:
                        # [batch, seq_len, channels] -> reshape to spatial grid
                        bs, seq_len, c = latent_tensor.shape
                        if seq_len == batch * t_len * h_len * w_len:
                            latent_tensor = latent_tensor.reshape(bs, t_len, h_len, w_len, c)
                            patch_variance = latent_tensor.var(dim=-1)  # [batch, t_len, h_len, w_len]
                        else:
                            print(f"[HunyuanSpatialWarp] WARNING: Cannot reshape latent tensor for variance computation")
                            patch_variance = None
                    else:
                        print(f"[HunyuanSpatialWarp] WARNING: Unexpected latent tensor shape: {latent_tensor.shape}")
                        patch_variance = None
                    
                    if patch_variance is not None:
                        # Normalize variance to [0, 1] range
                        # Convert to float32 for quantile computation (quantile requires float or double dtype)
                        patch_variance_float = patch_variance.float()
                        variance_min = patch_variance_float.min()
                        variance_max = patch_variance_float.max()
                        if variance_max > variance_min:
                            variance_norm = (patch_variance_float - variance_min) / (variance_max - variance_min)
                        else:
                            variance_norm = torch.zeros_like(patch_variance_float)
                        
                        # Threshold to find important regions (top 50% by variance - more lenient to preserve more content)
                        saliency_threshold = variance_norm.quantile(0.5)
                        important_mask = variance_norm > saliency_threshold  # [batch, t_len, h_len, w_len]
                        
                        if warp_mode == "saliency_preserve":
                            # Keep important regions' coordinates unchanged, warp background more gently
                            # For background: apply a gentler warp (partial mirror instead of full mirror)
                            # This preserves more of the original video structure
                            h_coords_warped = torch.where(
                                important_mask,
                                h_coords,  # Keep original for important regions
                                h_coords * 0.3 + (h_len - 1 - h_coords) * 0.7  # Gentle warp for background (70% mirror, 30% original)
                            )
                            w_coords_warped = torch.where(
                                important_mask,
                                w_coords,  # Keep original for important regions
                                w_coords * 0.3 + (w_len - 1 - w_coords) * 0.7  # Gentle warp for background (70% mirror, 30% original)
                            )
                            
                        elif warp_mode == "saliency_mirror":
                            # Mirror important regions, leave background static
                            h_coords_warped = torch.where(
                                important_mask,
                                h_len - 1 - h_coords,  # Mirror for important regions
                                h_coords  # Keep original for background
                            )
                            w_coords_warped = torch.where(
                                important_mask,
                                w_len - 1 - w_coords,  # Mirror for important regions
                                w_coords  # Keep original for background
                            )
                            
                        elif warp_mode == "saliency_swap":
                            # Swap coordinates between high-saliency regions
                            # Find all important region coordinates
                            important_h = h_coords[important_mask]
                            important_w = w_coords[important_mask]
                            
                            if len(important_h) > 1:
                                # Shuffle important region coordinates
                                perm = torch.randperm(len(important_h), device=important_h.device)
                                shuffled_h = important_h[perm]
                                shuffled_w = important_w[perm]
                                
                                # Create output coordinates
                                h_coords_warped = h_coords.clone()
                                w_coords_warped = w_coords.clone()
                                
                                # Assign shuffled coordinates to important regions
                                idx = 0
                                for b in range(batch):
                                    for t_idx in range(t_len):
                                        for h_idx in range(h_len):
                                            for w_idx in range(w_len):
                                                if important_mask[b, t_idx, h_idx, w_idx]:
                                                    h_coords_warped[b, t_idx, h_idx, w_idx] = shuffled_h[idx]
                                                    w_coords_warped[b, t_idx, h_idx, w_idx] = shuffled_w[idx]
                                                    idx += 1
                            else:
                                # Not enough important regions to swap, fall back to mirror
                                h_coords_warped = h_len - 1 - h_coords
                                w_coords_warped = w_len - 1 - w_coords
            
            # Attention-guided coordinate remapping modes
            elif warp_mode in ["attention_cluster_swap", "attention_mirror"]:
                if latent_tensor is None:
                    print(f"[HunyuanSpatialWarp] WARNING: {warp_mode} requires latent tensor, falling back to no warp")
                else:
                    # For attention-guided, we use feature similarity as a proxy for attention
                    # Compute patch features and find similar patches
                    if latent_tensor.dim() == 5:
                        # [batch, channels, t, h, w]
                        bs, c, t, h, w = latent_tensor.shape
                        if h != h_len or w != w_len:
                            latent_tensor = torch.nn.functional.interpolate(
                                latent_tensor.reshape(bs * t, c, h, w),
                                size=(h_len, w_len),
                                mode='bilinear',
                                align_corners=False
                            ).reshape(bs, t, c, h_len, w_len)
                            latent_tensor = latent_tensor.permute(0, 2, 1, 3, 4)  # [bs, c, t, h_len, w_len]
                        # Reshape to [batch, t_len, h_len, w_len, channels]
                        features = latent_tensor.permute(0, 2, 3, 4, 1).reshape(bs, t_len, h_len, w_len, c)
                    elif latent_tensor.dim() == 3:
                        # [batch, seq_len, channels]
                        bs, seq_len, c = latent_tensor.shape
                        if seq_len == bs * t_len * h_len * w_len:
                            features = latent_tensor.reshape(bs, t_len, h_len, w_len, c)
                        else:
                            print(f"[HunyuanSpatialWarp] WARNING: Cannot reshape latent tensor for attention computation")
                            features = None
                    else:
                        features = None
                    
                    if features is not None:
                        # Normalize features for similarity computation
                        features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)  # [batch, t_len, h_len, w_len, c]
                        
                        # Compute similarity matrix (cosine similarity between patches)
                        # Flatten spatial dimensions: [batch, t_len, h_len*w_len, c]
                        features_flat = features_norm.reshape(bs, t_len, h_len * w_len, c)
                        
                        # Compute cosine similarity: [batch, t_len, h_len*w_len, h_len*w_len]
                        similarity = torch.bmm(
                            features_flat.reshape(bs * t_len, h_len * w_len, c),
                            features_flat.reshape(bs * t_len, h_len * w_len, c).transpose(1, 2)
                        ).reshape(bs, t_len, h_len * w_len, h_len * w_len)
                        
                        # Find clusters: patches with high similarity (>0.7 threshold)
                        similarity_threshold = 0.7
                        clusters = similarity > similarity_threshold
                        
                        if warp_mode == "attention_cluster_swap":
                            # Swap coordinates within attention clusters
                            h_coords_flat = h_coords.reshape(bs, t_len, h_len * w_len)
                            w_coords_flat = w_coords.reshape(bs, t_len, h_len * w_len)
                            h_coords_warped_flat = h_coords_flat.clone()
                            w_coords_warped_flat = w_coords_flat.clone()
                            
                            for b in range(bs):
                                for t_idx in range(t_len):
                                    # Find clusters for this frame
                                    cluster_mask = clusters[b, t_idx]  # [h_len*w_len, h_len*w_len]
                                    # For each patch, find its cluster members
                                    for i in range(h_len * w_len):
                                        cluster_members = torch.where(cluster_mask[i])[0]
                                        if len(cluster_members) > 1:
                                            # Swap coordinates within cluster
                                            perm = torch.randperm(len(cluster_members), device=cluster_members.device)
                                            shuffled = cluster_members[perm]
                                            h_coords_warped_flat[b, t_idx, cluster_members] = h_coords_flat[b, t_idx, shuffled]
                                            w_coords_warped_flat[b, t_idx, cluster_members] = w_coords_flat[b, t_idx, shuffled]
                            
                            h_coords_warped = h_coords_warped_flat.reshape(bs, t_len, h_len, w_len)
                            w_coords_warped = w_coords_warped_flat.reshape(bs, t_len, h_len, w_len)
                            
                        elif warp_mode == "attention_mirror":
                            # Mirror coordinates within attention clusters
                            h_coords_flat = h_coords.reshape(bs, t_len, h_len * w_len)
                            w_coords_flat = w_coords.reshape(bs, t_len, h_len * w_len)
                            h_coords_warped_flat = h_coords_flat.clone()
                            w_coords_warped_flat = w_coords_flat.clone()
                            
                            for b in range(bs):
                                for t_idx in range(t_len):
                                    cluster_mask = clusters[b, t_idx]
                                    for i in range(h_len * w_len):
                                        cluster_members = torch.where(cluster_mask[i])[0]
                                        if len(cluster_members) > 1:
                                            # Mirror coordinates within cluster
                                            h_center = h_coords_flat[b, t_idx, cluster_members].mean()
                                            w_center = w_coords_flat[b, t_idx, cluster_members].mean()
                                            h_coords_warped_flat[b, t_idx, cluster_members] = 2 * h_center - h_coords_flat[b, t_idx, cluster_members]
                                            w_coords_warped_flat[b, t_idx, cluster_members] = 2 * w_center - w_coords_flat[b, t_idx, cluster_members]
                            
                            h_coords_warped = h_coords_warped_flat.reshape(bs, t_len, h_len, w_len)
                            w_coords_warped = w_coords_warped_flat.reshape(bs, t_len, h_len, w_len)
            
            # Blend with original based on strength
            # This applies uniformly to ALL modes, ensuring consistent strength behavior
            # Formula: result = original + (warped - original) * strength
            # - strength = 0.0: no warping (result = original)
            # - strength = 1.0: full warping (result = warped)
            # - strength = 0.5: half warping (result = halfway between original and warped)
            if warp_strength <= 0.0:
                # No warping - use original coordinates
                h_coords_warped = h_coords
                w_coords_warped = w_coords
            elif warp_strength >= 1.0:
                # Full warping - use warped coordinates as-is (no blending needed)
                pass  # h_coords_warped and w_coords_warped already contain the warped values
            else:
                # Partial warping - blend between original and warped
                h_coords_warped = h_coords + (h_coords_warped - h_coords) * warp_strength
                w_coords_warped = w_coords + (w_coords_warped - w_coords) * warp_strength
            
            # Clamp coordinates to valid range
            h_coords_warped = torch.clamp(h_coords_warped, 0, h_len - 1)
            w_coords_warped = torch.clamp(w_coords_warped, 0, w_len - 1)
            
            # Update img_ids with warped coordinates
            img_ids_warped = img_ids_grid.clone()
            img_ids_warped[:, :, :, :, 1] = h_coords_warped
            img_ids_warped[:, :, :, :, 2] = w_coords_warped
            
            # Reshape back to [batch, seq_len, 3]
            return img_ids_warped.reshape(batch, seq_len, 3)

        # Check if model has forward_orig and patch it directly
        if hasattr(diffusion_model, 'forward_orig'):
            original_forward_orig = diffusion_model.forward_orig
            
            def tweaked_forward_orig(img, img_ids, txt, txt_ids, txt_mask, timesteps, y, *args, **kwargs):
                # Extract parameters from kwargs
                control = kwargs.pop('control', None)
                transformer_options = kwargs.pop('transformer_options', {})
                
                # Get warp parameters from transformer_options or use closure values
                warp_mode = transformer_options.get('spatial_warp_mode', mode)
                warp_strength = transformer_options.get('spatial_warp_strength', strength)
                
                # Infer spatial dimensions from img_ids
                # img_ids shape: [batch, seq_len, 3]
                batch, seq_len, _ = img_ids.shape
                
                # We need to infer h_len and w_len from the actual coordinate values
                # The coordinates should be in a grid pattern
                # Try to infer from the unique values in height/width coordinates
                h_coords_unique = torch.unique(img_ids[:, :, 1]).sort()[0]
                w_coords_unique = torch.unique(img_ids[:, :, 2]).sort()[0]
                
                h_len = len(h_coords_unique)
                w_len = len(w_coords_unique)
                t_len = seq_len // (h_len * w_len)
                
                if t_len * h_len * w_len == seq_len and warp_strength > 0:
                    # Apply spatial warping to img_ids
                    # For forward_orig, img is in [batch, channels, t, h, w] format (5D) before img_in processing
                    # We can use it directly for variance/attention-based warping
                    latent_for_warp = None
                    if img is not None:
                        if img.dim() == 5:
                            # img is in [batch, channels, t, h, w] format - perfect for our warping
                            latent_for_warp = img
                        elif img.dim() == 3 and img.shape[1] == seq_len:
                            # img is in sequence format [batch, seq_len, channels] (after img_in)
                            latent_for_warp = img
                    img_ids = warp_spatial_coords(img_ids, h_len, w_len, warp_mode, warp_strength, latent_tensor=latent_for_warp)
                
                # Store in transformer_options if requested
                if transformer_options.get('expose_internals', False):
                    transformer_options['hunyuan_internals'] = {
                        'img_ids': img_ids,
                        'img': img,
                        'txt': txt,
                    }
                
                # Call original forward_orig with modified img_ids
                return original_forward_orig(img, img_ids, txt, txt_ids, txt_mask, timesteps, y, *args, control=control, transformer_options=transformer_options, **kwargs)
            
            # Patch forward_orig directly
            diffusion_model.forward_orig = tweaked_forward_orig
        else:
            # For base HunyuanVideo without forward_orig, patch forward method
            original_forward = diffusion_model.forward
            
            def tweaked_forward(x, timestep, context, y, guidance, attention_mask=None, control=None, transformer_options={}, **kwargs):
                # Get warp parameters from transformer_options or use closure values
                warp_mode = transformer_options.get('spatial_warp_mode', mode)
                warp_strength = transformer_options.get('spatial_warp_strength', strength)
                
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
                
                # Apply spatial warping to img_ids
                if warp_strength > 0:
                    # Pass x (latent tensor) for variance/attention-based warping
                    img_ids = warp_spatial_coords(img_ids, h_len, w_len, warp_mode, warp_strength, latent_tensor=x)
                
                # Store in transformer_options if requested
                if transformer_options.get('expose_internals', False):
                    transformer_options['hunyuan_internals'] = {
                        'img_ids': img_ids,
                        'x': x,
                        'context': context,
                    }
                
                # Pass through with transformer_options
                transformer_options['spatial_warp_mode'] = warp_mode
                transformer_options['spatial_warp_strength'] = warp_strength
                
                return original_forward(x, timestep, context, y, guidance, attention_mask, control, transformer_options, **kwargs)
            
            # Patch the forward method
            diffusion_model.forward = tweaked_forward
        
        # Store parameters in model_options for access during forward pass
        model_options = m.model_options.copy()
        transformer_options = model_options.get('transformer_options', {}).copy()
        
        transformer_options['spatial_warp_mode'] = mode
        transformer_options['spatial_warp_strength'] = strength
        
        model_options['transformer_options'] = transformer_options
        m.model_options = model_options
        
        # Mark as patched
        setattr(diffusion_model, "_vecnode_spatial_warp_patched", True)
        
        print("[HunyuanSpatialWarp] Patch complete. Spatial warping will be applied via img_ids modification.")
        
        return (m,)



