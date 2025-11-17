import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io
import torch  # ← Critical for correct tensor return

BASE_URL = "https://api.topazlabs.com/image/v1"

# Modes
TOPAZ_MODES = ["enhance", "sharpen", "denoise", "restore", "lighting"]

# Updated & accurate model lists (November 2025)
ENHANCE_GAN_MODELS = ["Standard V2", "Low Resolution V2", "CGI", "High Fidelity V2", "Text Refine"]
ENHANCE_GEN_MODELS = ["Redefine", "Recovery V2", "Standard MAX", "Wonder"]
SHARPEN_GAN_MODELS = ["Standard", "Strong", "Lens Blur", "Lens Blur V2", "Motion Blur", "Natural", "Refocus"]
SHARPEN_GEN_MODELS = ["Super Focus V2"]
DENOISE_GAN_MODELS = ["Normal", "Strong", "Extreme"]
RESTORE_GEN_MODELS = ["Dust-Scratch"]
LIGHTING_GAN_MODELS = ["Adjust", "White Balance"]

TOPAZ_MODELS = list(set(
    ENHANCE_GAN_MODELS + ENHANCE_GEN_MODELS +
    SHARPEN_GAN_MODELS + SHARPEN_GEN_MODELS +
    DENOISE_GAN_MODELS + RESTORE_GEN_MODELS + LIGHTING_GAN_MODELS
))

FORMAT_ACCEPT = {"jpeg": "image/jpeg", "png": "image/png", "tiff": "image/tiff"}
FORMAT_MAGIC = {
    "jpeg": b'\xff\xd8\xff',
    "png": b'\x89PNG\r\n\x1a\n',
    "tiff": [b'II*\x00', b'MM*\x00']
}

class TopazUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Topaz API key"
                }),
                "mode": (TOPAZ_MODES, {"default": "enhance"}),
                "model": (TOPAZ_MODELS, {"default": "Standard V2"}),
                "scale_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Set >1.0 to auto-upscale. Set exactly 1.0 to use manual width/height."
                }),
            },
            "optional": {
                "output_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32000,
                    "step": 64,
                    "display": "number",
                    "visible": {"condition": {"scale_multiplier": ["1.0"]}, "value": True},
                    "tooltip": "Only used when scale_multiplier = 1.0"
                }),
                "output_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32000,
                    "step": 64,
                    "display": "number",
                    "visible": {"condition": {"scale_multiplier": ["1.0"]}, "value": True},
                    "tooltip": "Only used when scale_multiplier = 1.0"
                }),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png", "tiff"], {"default": "jpeg"}),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpen_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fix_compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "timeout_seconds": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "image/topaz"

    def _get_submit_path(self, mode, model):
        if mode == "enhance":
            if model in ENHANCE_GEN_MODELS: return "/enhance-gen/async"
            if model in ENHANCE_GAN_MODELS: return "/enhance/async"
        elif mode == "sharpen":
            if model in SHARPEN_GEN_MODELS: return "/sharpen-gen/async"
            if model in SHARPEN_GAN_MODELS: return "/sharpen/async"
        elif mode == "denoise":
            if model in DENOISE_GAN_MODELS: return "/denoise/async"
        elif mode == "restore":
            if model in RESTORE_GEN_MODELS: return "/restore-gen/async"
        elif mode == "lighting":
            if model in LIGHTING_GAN_MODELS: return "/lighting/async"
        raise ValueError(f"Invalid combination: mode='{mode}', model='{model}'")

    def _submit_job(self, image_path, api_key, mode, model, params):
        path = self._get_submit_path(mode, model)
        headers = {"X-API-Key": api_key, "Accept": "application/json"}
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f)}
            data = {k: str(v) for k, v in params.items()}
            response = requests.post(f"{BASE_URL}{path}", headers=headers, files=files, data=data, timeout=60)
        response.raise_for_status()
        resp_json = response.json()
        print(f"[Topaz] Submit: {resp_json}")

        process_id = resp_json.get("process_id") or response.headers.get("X-Process-ID")
        if not process_id:
            raise ValueError("No process_id returned")
        return process_id, 0

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        url = f"{BASE_URL}/status/{process_id}"
        print(f"[Topaz] Status URL: {url}")

        while time.time() - start < timeout:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 404:
                    time.sleep(5); continue
                resp.raise_for_status()
                status = resp.json()
                st = status.get("status", "").strip().lower()
                prog = status.get("progress", "N/A")
                print(f"[Topaz] Status: {st.capitalize()} | Progress: {prog}%")

                if st == "completed":
                    print("[Topaz] Job completed!")
                    return True
                if st == "failed":
                    raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[Topaz] Status check error: {e}")
            time.sleep(8)
        raise TimeoutError(f"Timed out after {timeout}s")

    def _download_result(self, process_id, api_key, output_format):
        headers = {"X-API-Key": api_key, "Accept": "application/json"}
        url = f"{BASE_URL}/download/{process_id}"
        print(f"[Topaz] Download URL: {url}")

        for attempt in range(8):
            resp = requests.get(url, headers=headers, timeout=30)
            print(f"[Topaz] Download attempt {attempt+1} → {resp.status_code}")

            if resp.status_code == 409:
                time.sleep(3); continue

            resp.raise_for_status()
            data = resp.json()
            dl_url = data.get("download_url")
            if not dl_url:
                raise ValueError("No download_url received")

            img_resp = requests.get(dl_url, timeout=60)
            img_resp.raise_for_status()
            content = img_resp.content

            # Validate format
            if output_format == "jpeg" and content.startswith(FORMAT_MAGIC["jpeg"]):
                pass
            elif output_format == "png" and content.startswith(FORMAT_MAGIC["png"]):
                pass
            elif output_format == "tiff" and content[:4] in FORMAT_MAGIC["tiff"]:
                pass
            else:
                print(f"[Topaz] Invalid image content → retrying...")
                time.sleep(3)
                continue

            print(f"[Topaz] Valid image received ({len(content)/1024:.1f} KB)")
            return content

        raise Exception("Failed to download valid image after retries")

    def process(self, image, api_key, mode, model,
                scale_multiplier=1.0, output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5, strength=0.5,
                fix_compression=0.0, timeout_seconds=300):

        if not api_key.strip():
            raise ValueError("Topaz API key required")

        h, w = image.shape[1], image.shape[2]

        # Auto-scale logic
        if scale_multiplier > 1.0:
            if output_width > 0 or output_height > 0:
                print(f"[Topaz] Ignoring manual width/height → using scale_multiplier ×{scale_multiplier}")
            new_w = max(64, min(32000, int(w * scale_multiplier)))
            new_h = max(64, min(32000, int(h * scale_multiplier)))
            output_width, output_height = new_w, new_h
            print(f"[Topaz] Auto-scale ×{scale_multiplier}: {w}×{h} → {new_w}×{new_h}")
        else:
            if mode == "enhance" and output_width == 0 and output_height == 0:
                print("[Topaz] No output size specified → using original dimensions")

        params = {
            "model": model,
            "output_format": output_format,
            "face_enhancement": str(face_enhancement).lower(),
            "denoise_strength": denoise_strength,
            "sharpen_strength": sharpen_strength,
            "strength": strength,
            "fix_compression": fix_compression,
        }
        if mode == "enhance":
            if output_width: params["output_width"] = output_width
            if output_height: params["output_height"] = output_height
            params["crop_to_fill"] = str(crop_to_fill).lower()

        # Save input image
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        save_args = {"quality": 95} if output_format == "jpeg" else {}
        if output_format == "png":
            save_args["compress_level"] = 4
        elif output_format == "tiff":
            save_args["compression"] = "tiff_deflate"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            pil_img.save(tmp.name, format=output_format.upper(), **save_args)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting {mode} | {model}")
            process_id, _ = self._submit_job(input_path, api_key, mode, model, params)

            print(f"[Topaz] Waiting for completion...")
            self._wait_for_completion(process_id, api_key, timeout_seconds)

            print(f"[Topaz] Downloading result...")
            result_bytes = self._download_result(process_id, api_key, output_format)

            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if result_np.ndim == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]

            # Critical: return PyTorch tensor, not numpy
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)

            print(f"[Topaz] Success! Output: {result_tensor.shape} | {result_pil.format}")
            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler (API)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
