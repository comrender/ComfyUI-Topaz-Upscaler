import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io
import math

BASE_URL = "https://api.topazlabs.com/image/v1"

# === ALL MODES ===
TOPAZ_MODES = ["enhance", "sharpen", "denoise", "restore", "lighting"]

# === ALL MODELS (from official docs) ===
TOPAZ_MODELS = [
    # Enhance - Standard
    "Standard V2", "Low Resolution V2", "CGI", "High Fidelity V2", "Text Refine",
    # Enhance - Generative
    "Redefine", "Recovery V2", "Standard MAX", "Wonder",
    # Sharpen - Standard
    "Standard", "Strong", "Lens Blur", "Lens Blur V2", "Motion Blur", "Natural", "Refocus",
    # Sharpen - Generative
    "Super Focus V2",
    # Denoise
    "Normal", "Strong", "Extreme",
    # Restore
    "Dust-Scratch",
    # Lighting
    "Adjust", "White Balance"
]

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
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "display": "slider"
                }),
                "output_width": ("INT", {
                    "default": 0, "min": 1, "max": 32000, "step": 64,
                    "display": "number"
                }),
                "output_height": ("INT", {
                    "default": 0, "min": 1, "max": 32000, "step": 64,
                    "display": "number"
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

    def _get_endpoint_path(self, mode):
        paths = {
            "enhance": "/enhance/async",
            "sharpen": "/sharpen/async",
            "denoise": "/denoise/async",
            "restore": "/restore-gen/async",
            "lighting": "/lighting/async"
        }
        return paths[mode]

    def _submit_job(self, image_path, api_key, mode, params):
        headers = {"X-API-Key": api_key, "accept": "application/json"}
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}{self._get_endpoint_path(mode)}",
                                   headers=headers, files=files, data=params)
        response.raise_for_status()
        data = response.json()
        
        # Log full response for debugging
        print(f"[Topaz] Submit response: {data}")
        
        # Extract process_id (standard from docs)
        process_id = data.get("process_id")
        if not process_id:
            raise ValueError(f"No process_id in response: {data}")
        
        eta = data.get("eta", 0)
        print(f"[Topaz] Process ID: {process_id} | ETA: {eta}")
        
        return process_id, eta

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        retries = 0
        max_retries = 3
        
        status_url = f"{BASE_URL}/status/{process_id}"
        print(f"[Topaz] Status URL: {status_url}")
        
        while time.time() - start < timeout:
            try:
                resp = requests.get(status_url, headers=headers)
                
                if resp.status_code == 404:
                    raise Exception(f"Process not found (404). Check if job started correctly or API key has access. URL: {status_url}")
                
                resp.raise_for_status()
                status = resp.json()
                
                print(f"[Topaz] Status: {status.get('status', 'unknown')}")
                
                if status.get("status") == "completed":
                    return True
                if status.get("status") == "failed":
                    raise Exception(f"Topaz job failed: {status.get('error', 'Unknown error')}")
                
                retries = 0  # Reset on success
                
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Status check failed after {max_retries} retries: {e}")
                print(f"[Topaz] Status retry {retries}/{max_retries}: {e}")
                time.sleep(2)  # Short backoff
            
            time.sleep(5)
        
        raise TimeoutError(f"Job {process_id} timed out after {timeout}s. Try increasing timeout.")

    def _download_result(self, process_id, api_key, output_format):
        accept = FORMAT_ACCEPT.get(output_format, "image/jpeg")
        headers = {"X-API-Key": api_key, "accept": accept}
        
        download_url = f"{BASE_URL}/download/{process_id}"
        print(f"[Topaz] Download URL: {download_url}")
        
        resp = requests.get(download_url, headers=headers)
        
        if resp.status_code == 404:
            raise Exception(f"Download not ready or not found (404). Ensure job completed. URL: {download_url}")
        
        resp.raise_for_status()
        return resp.content

    def process(self, image, api_key, mode, model,
                scale_multiplier=1.0,
                output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5,
                strength=0.5, fix_compression=0.0, timeout_seconds=300):

        if not api_key.strip():
            raise ValueError("Topaz API key is required.")

        # === INPUT IMAGE SIZE ===
        h, w = image.shape[1], image.shape[2]  # HWC

        # === AUTO-CALCULATE WIDTH/HEIGHT FROM MULTIPLIER (ENHANCE ONLY) ===
        use_auto_size = (mode == "enhance") and (scale_multiplier != 1.0)
        if use_auto_size:
            new_w = int(round(w * scale_multiplier))
            new_h = int(round(h * scale_multiplier))
            # Clamp to API limits
            new_w = min(max(new_w, 1), 32000)
            new_h = min(max(new_h, 1), 32000)
            output_width, output_height = new_w, new_h
            print(f"[Topaz] Scale ×{scale_multiplier} → {w}×{h} → {new_w}×{new_h}")

        # === BUILD PARAMS ===
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

        # === SAVE INPUT TO TEMP FILE ===
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            save_kwargs = {}
            if output_format == "jpeg":
                save_kwargs["quality"] = 95
            pil_img.save(tmp.name, format=output_format.upper(), **save_kwargs)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting {mode} job | Model: {model} | Size: {output_width}×{output_height}")
            process_id, eta = self._submit_job(input_path, api_key, mode, params)

            print(f"[Topaz] Waiting for completion...")
            self._wait_for_completion(process_id, api_key, timeout_seconds)

            print(f"[Topaz] Downloading result...")
            result_bytes = self._download_result(process_id, api_key, output_format)

            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]  # Drop alpha
            result_tensor = result_np[np.newaxis, ...]

            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
