import torch

class HuMoLipsyncSuppress:
    """
    Simplified HuMo audio editor that suppresses lip-sync on boolean trigger.
    When enabled=True, applies preset band gains to reduce mouth movement.
    When enabled=False, passes through unchanged.
    
    Preset settings (from reference image):
      - gain_b0: 4.00 (shallow edges/onsets)
      - gain_b1: 4.00 (short-term rhythm)
      - gain_b2: 0.50 (phrase patterns)
      - gain_b3: 0.01 (long-range cadence)
      - gain_b4: 0.01 (top semantic)
      - ema_beta: 0.90
      - preserve_rms: False
      - alpha_mix: 1.00
      - global_gain: 1.00
      - clamp_std: 0.00 (disabled)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable lipsync suppression (apply preset gains)"
                }),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "apply"
    CATEGORY = "HuMo Audio/Motion"
    DESCRIPTION = "Apply preset band gains to suppress lip-sync motion (boolean trigger)."

    # Preset values from reference image
    PRESET_GAINS = [4.00, 4.00, 0.50, 0.01, 0.01]
    PRESET_EMA_BETA = 0.90
    PRESET_PRESERVE_RMS = False
    PRESET_ALPHA_MIX = 1.00
    PRESET_GLOBAL_GAIN = 1.00
    PRESET_CLAMP_STD = 0.00

    def _rms(self, x, eps=1e-6):
        """Per-frame, per-band RMS (keep channel dim)"""
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    def apply(self, image_embeds, enabled):
        # If disabled, return unchanged
        if not enabled:
            return (image_embeds,)

        # Shallow copy dict to avoid side-effects
        embeds = dict(image_embeds)
        key = "humo_audio_emb"
        
        if key not in embeds:
            raise ValueError("Missing key 'humo_audio_emb' in WANVIDIMAGE_EMBEDS")

        x = embeds[key]  # [T, 5, 1280]
        if x.ndim != 3 or x.shape[1] != 5:
            raise ValueError(f"humo_audio_emb expected shape [T,5,C], got {tuple(x.shape)}")

        device = x.device
        dtype = x.dtype
        x_orig = x

        # Apply preset per-band gains
        gains = torch.tensor(self.PRESET_GAINS, device=device, dtype=dtype).view(1, 5, 1)

        # Temporal EMA smoothing
        if x.shape[0] > 1 and self.PRESET_EMA_BETA > 0.0:
            s = x[0].clone()
            out_seq = [s]
            beta = torch.tensor(self.PRESET_EMA_BETA, device=device, dtype=dtype)
            one_m = (1.0 - beta).to(dtype)
            for t in range(1, x.shape[0]):
                s = beta * s + one_m * x[t]
                out_seq.append(s)
            x_smooth = torch.stack(out_seq, dim=0)
        else:
            x_smooth = x

        # Apply gains
        x_edit = x_smooth * gains

        # RMS preservation (disabled in preset)
        if self.PRESET_PRESERVE_RMS:
            rms_orig = self._rms(x_orig)
            rms_edit = self._rms(x_edit)
            x_edit = x_edit * (rms_orig / rms_edit)

        # Residual blend (alpha=1.0 means full edit)
        alpha = torch.tensor(self.PRESET_ALPHA_MIX, device=device, dtype=dtype)
        x_mixed = (1.0 - alpha) * x_orig + alpha * x_edit
        
        # Global gain
        if self.PRESET_GLOBAL_GAIN != 1.0:
            x_mixed = x_mixed * self.PRESET_GLOBAL_GAIN

        # Clamp (disabled in preset)
        if self.PRESET_CLAMP_STD > 0.0:
            mean = x_mixed.mean()
            std = x_mixed.std().clamp_min(1e-6)
            lo = mean - self.PRESET_CLAMP_STD * std
            hi = mean + self.PRESET_CLAMP_STD * std
            x_mixed = x_mixed.clamp(lo, hi)

        embeds[key] = x_mixed
        return (embeds,)


NODE_CLASS_MAPPINGS = {
    "HuMoLipsyncSuppress": HuMoLipsyncSuppress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoLipsyncSuppress": "HuMo Lipsync Suppress",
}
