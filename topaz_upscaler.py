import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io

BASE_URL = "https://api.topazlabs.com/image/v1"

# Modes
TOPAZ_MODES = ["enhance", "sharpen", "denoise", "restore", "lighting"]

# Models (full list)
TOPAZ_MODELS_STANDARD = [
    "Standard V2", "Low Resolution V2", "CGI", "High Fidelity V2", "Text Refine",
    "Standard", "Strong", "Lens Blur", "Lens Blur V2", "Motion Blur", "Natural", "Refocus",
    "Normal", "Extreme", "Dust-Scratch", "Adjust", "White Balance"
]
TOPAZ_MODELS_GENERATIVE = [
    "Redefine", "Recovery V2", "Standard MAX", "Wonder", "Super Focus V2"
]
TOPAZ_MODELS = TOPAZ_MODELS_STANDARD + TOPAZ_MODELS_GENERATIVE

# Accept headers
FORMAT_ACCEPT = {"jpeg": "image/jpeg", "png": "image/png", "tiff": "image/tiff"}

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
            },
            "optional": {
                "scale_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1
                }),
                "output_width": ("INT", {
                    "default": 0, "min": 1, "max": 32000, "step": 64
                }),
                "output_height": ("INT", {
                    "default": 0, "min": 1, "max": 32000, "step": 64
                }),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png", "tiff"], {"default": "jpeg"}),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpen_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fix_compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "timeout_seconds": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 60}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "image/topaz"

    def _get_submit_path(self, mode, model):
        paths = {
            "enhance": "/enhance/async" if model in TOPAZ_MODELS_GENERATIVE else "/enhance/async",
            "sharpen": "/sharpen/async",
            "denoise": "/denoise/async",
            "restore": "/restore-gen/async",
            "lighting": "/lighting/async"
        }
        return paths.get(mode, "/enhance/async")

    def _submit_job(self, image_path, api_key, mode, model, params):
        path = self._get_submit_path(mode, model)
        headers = {"X-API-Key": api_key, "Accept": "application/json"}
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            # All params as STRINGS
            data = {k: str(v) for k, v in params.items()}
            response = requests.post(f"{BASE_URL}{path}", headers=headers, files=files, data=data)
        response.raise_for_status()
        data = response.json()
        print(f"[Topaz] Submit response: {data}")
        
        process_id = data.get("process_id") or response.headers.get("X-Process-ID")
        if not process_id:
            raise ValueError(f"No process_id in response: {data}")
        
        eta = int(response.headers.get("X-ETA", 0)) if response.headers.get("X-ETA") else 0
        eta_remaining = max(eta - int(time.time()), 0) if eta else 0
        if eta_remaining > 0:
            print(f"[Topaz] ETA remaining: ~{eta_remaining}s")
        
        return process_id, eta_remaining

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        poll_interval = 10
        
        status_url = f"{BASE_URL}/status/{process_id}"
        print(f"[Topaz] Status URL: {status_url}")
        
        while time.time() - start < timeout:
            resp = requests.get(status_url, headers=headers)
            print(f"[Topaz] Status code: {resp.status_code}")
            
            if resp.status_code == 404:
                print("[Topaz] 404 - job pending. Retrying...")
                time.sleep(poll_interval)
                continue
            
            resp.raise_for_status()
            status = resp.json()
            status_val = status.get("status", "").strip()  # Get raw status
            progress = status.get('progress', 'N/A')
            print(f"[Topaz] Status: {status_val} | Progress: {progress}%")
            
            # FIXED: Case-insensitive check + lowercase for comparison
            if status_val.lower() == "completed":
                print("[Topaz] Job completed! Proceeding to download.")
                return True
            if status_val.lower() == "failed":
                raise Exception(f"Job failed: {status.get('error', 'Unknown')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Timeout after {timeout}s. Check dashboard with process_id: {process_id}")

    def _download_result(self, process_id, api_key, output_format):
        accept = FORMAT_ACCEPT.get(output_format, "image/jpeg")
        headers = {"X-API-Key": api_key, "Accept": accept}
        
        download_url = f"{BASE_URL}/download/{process_id}"
        print(f"[Topaz] Download URL: {download_url} (Format: {output_format})")
        
        resp = requests.get(download_url, headers=headers)
        print(f"[Topaz] Download code: {resp.status_code}")
        
        if resp.status_code == 404:
            raise Exception(f"Download 404 - job incomplete. ID: {process_id}")
        
        resp.raise_for_status()
        return resp.content

    def process(self, image, api_key, mode, model,
                scale_multiplier=1.0,
                output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5,
                strength=0.5, fix_compression=0.0, timeout_seconds=300):

        if not api_key.strip():
            raise ValueError("Topaz API key required.")

        # Input size
        h, w = image.shape[1], image.shape[2]

        # Auto-scale (enhance only)
        use_auto = (mode == "enhance") and (scale_multiplier != 1.0)
        if use_auto:
            new_w = min(max(int(w * scale_multiplier), 1), 32000)
            new_h = min(max(int(h * scale_multiplier), 1), 32000)
            output_width, output_height = new_w, new_h
            print(f"[Topaz] Auto-scale ×{scale_multiplier}: {w}×{h} → {new_w}×{new_h}")

        # Params
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
            if output_width > 0:
                params["output_width"] = output_width
            if output_height > 0:
                params["output_height"] = output_height
            params["crop_to_fill"] = str(crop_to_fill).lower()

        # Temp input
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            kwargs = {"quality": 95} if output_format == "jpeg" else {}
            pil_img.save(tmp.name, format=output_format.upper(), **kwargs)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting {mode} | Model: {model} | Size: {output_width}×{output_height}")
            process_id, eta_remaining = self._submit_job(input_path, api_key, mode, model, params)
            
            # Auto-extend if ETA > timeout
            if eta_remaining > timeout_seconds:
                timeout_seconds = eta_remaining + 60
                print(f"[Topaz] Extended timeout to {timeout_seconds}s based on ETA")

            print(f"[Topaz] Waiting...")
            self._wait_for_completion(process_id, api_key, timeout_seconds)

            print(f"[Topaz] Downloading...")
            result_bytes = self._download_result(process_id, api_key, output_format)

            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]  # RGB
            result_tensor = result_np[np.newaxis, ...]

            print(f"[Topaz] Success! Shape: {result_tensor.shape}")
            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
