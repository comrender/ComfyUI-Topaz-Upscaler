import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io
import math

BASE_URL = "https://api.topazlabs.com/image/v1"

# === MODES ===
TOPAZ_MODES = ["enhance", "sharpen", "denoise", "restore", "lighting"]

# === MODELS BY CATEGORY (from docs) ===
TOPAZ_MODELS_ENHANCE_STANDARD = [
    "Standard V2", "Low Resolution V2", "CGI", "High Fidelity V2", "Text Refine"
]
TOPAZ_MODELS_ENHANCE_GEN = [
    "Redefine", "Recovery V2", "Standard MAX", "Wonder"
]
TOPAZ_MODELS_SHARPEN_STANDARD = [
    "Standard", "Strong", "Lens Blur", "Lens Blur V2", "Motion Blur", "Natural", "Refocus"
]
TOPAZ_MODELS_SHARPEN_GEN = ["Super Focus V2"]
TOPAZ_MODELS_DENOISE = ["Normal", "Strong", "Extreme"]
TOPAZ_MODELS_RESTORE = ["Dust-Scratch"]
TOPAZ_MODELS_LIGHTING = ["Adjust", "White Balance"]

# All models for dropdown (flattened)
TOPAZ_MODELS = (TOPAZ_MODELS_ENHANCE_STANDARD + TOPAZ_MODELS_ENHANCE_GEN +
                TOPAZ_MODELS_SHARPEN_STANDARD + TOPAZ_MODELS_SHARPEN_GEN +
                TOPAZ_MODELS_DENOISE + TOPAZ_MODELS_RESTORE + TOPAZ_MODELS_LIGHTING)

# Accept header per format
FORMAT_ACCEPT = {"jpeg": "image/jpeg", "png": "image/png", "tiff": "image/tiff"}

class TopazUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "", "placeholder": "Enter your Topaz API key"
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
                "timeout_seconds": ("INT", {"default": 600, "min": 60, "max": 1800, "step": 60}),  # Increased default
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "image/topaz"

    def _get_endpoint_path(self, mode, model):
        paths = {
            "enhance": "/enhance/async",
            "sharpen": "/sharpen/async",
            "denoise": "/denoise/async",
            "restore": "/restore-gen/async",
            "lighting": "/lighting/async"
        }
        # Override for generative enhance
        if mode == "enhance" and model in TOPAZ_MODELS_ENHANCE_GEN:
            return "/enhance-gen/async"
        return paths[mode]

    def _submit_job(self, image_path, api_key, mode, model, params):
        path = self._get_endpoint_path(mode, model)
        headers = {"X-API-Key": api_key, "accept": "application/json"}
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}{path}", headers=headers, files=files, data=params)
        response.raise_for_status()
        data = response.json()
        
        # Also check headers for ID/ETA
        process_id = data.get("process_id") or response.headers.get("X-Process-ID")
        eta = data.get("eta") or int(response.headers.get("X-ETA", 0))
        
        print(f"[Topaz] Submit response: {data}")
        print(f"[Topaz] Headers ID: {response.headers.get('X-Process-ID')}, ETA: {eta}")
        
        if not process_id:
            raise ValueError(f"No process_id found: {data}")
        
        # Convert ETA to readable
        eta_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta)) if eta else "Unknown"
        print(f"[Topaz] Process ID: {process_id} | ETA: {eta_time}")
        
        return process_id, eta

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        retries = 0
        max_retries = 5  # Increased
        
        status_url = f"{BASE_URL}/getStatus?process_id={process_id}"
        print(f"[Topaz] Status URL: {status_url}")
        
        while time.time() - start < timeout:
            try:
                resp = requests.get(status_url, headers=headers)
                print(f"[Topaz] Status response code: {resp.status_code}")
                
                if resp.status_code == 404:
                    print("[Topaz] 404 on status - job may not be ready yet or endpoint mismatch")
                    time.sleep(10)  # Longer wait on 404
                    continue
                
                resp.raise_for_status()
                status = resp.json()
                
                print(f"[Topaz] Status: {status.get('status', 'unknown')} | Progress: {status.get('progress', 'N/A')}%")
                
                if status.get("status") == "completed":
                    return True
                if status.get("status") == "failed":
                    raise Exception(f"Job failed: {status.get('error', 'Unknown')}")
                
                retries = 0
                
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Status failed after {max_retries} retries: {e}")
                print(f"[Topaz] Status retry {retries}/{max_retries}: {e}")
                time.sleep(2 ** retries)  # Exponential backoff
            
            time.sleep(10)  # Poll every 10s (slower for generative)
        
        raise TimeoutError(f"Job {process_id} timed out after {timeout}s. Credits charged - check Topaz dashboard for manual download.")

    def _download_result(self, process_id, api_key, output_format):
        accept = FORMAT_ACCEPT.get(output_format, "image/jpeg")
        headers = {"X-API-Key": api_key, "accept": accept}
        
        download_url = f"{BASE_URL}/getDownloadOutput?process_id={process_id}"
        print(f"[Topaz] Download URL: {download_url} (Accept: {accept})")
        
        resp = requests.get(download_url, headers=headers)
        print(f"[Topaz] Download response code: {resp.status_code}")
        
        if resp.status_code == 404:
            raise Exception(f"Download 404 - job may not be complete. Check status manually with process_id: {process_id}")
        
        resp.raise_for_status()
        return resp.content

    def process(self, image, api_key, mode, model,
                scale_multiplier=1.0,
                output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5,
                strength=0.5, fix_compression=0.0, timeout_seconds=600):

        if not api_key.strip():
            raise ValueError("Topaz API key is required.")

        # === INPUT SIZE ===
        h, w = image.shape[1], image.shape[2]

        # === AUTO-SCALE ===
        use_auto_size = (mode == "enhance") and (scale_multiplier != 1.0)
        if use_auto_size:
            new_w = int(round(w * scale_multiplier))
            new_h = int(round(h * scale_multiplier))
            new_w = min(max(new_w, 1), 32000)
            new_h = min(max(new_h, 1), 32000)
            output_width, output_height = new_w, new_h
            print(f"[Topaz] Scale ×{scale_multiplier} → {w}×{h} → {new_w}×{new_h}")

        # === PARAMS ===
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

        # === TEMP FILE ===
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            save_kwargs = {"quality": 95} if output_format == "jpeg" else {}
            pil_img.save(tmp.name, format=output_format.upper(), **save_kwargs)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting {mode} job | Model: {model} | Size: {output_width}×{output_height}")
            process_id, eta = self._submit_job(input_path, api_key, mode, model, params)

            print(f"[Topaz] Waiting...")
            self._wait_for_completion(process_id, api_key, timeout_seconds)

            print(f"[Topaz] Fetching image...")
            result_bytes = self._download_result(process_id, api_key, output_format)

            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]
            result_tensor = result_np[np.newaxis, ...]

            print(f"[Topaz] Success! Output shape: {result_tensor.shape}")
            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
