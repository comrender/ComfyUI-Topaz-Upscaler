import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io

BASE_URL = "https://api.topazlabs.com/image/v1"

# All available modes/endpoints
TOPAZ_MODES = [
    "enhance",
    "sharpen",
    "denoise",
    "restore",
    "lighting"
]

# All available models (full list from API docs)
TOPAZ_MODELS = [
    # Enhance - Standard
    "Standard V2",
    "Low Resolution V2",
    "CGI",
    "High Fidelity V2",
    "Text Refine",
    # Enhance - Generative
    "Redefine",
    "Recovery V2",
    "Standard MAX",
    "Wonder",
    # Sharpen - Standard
    "Standard",  # Sharpen
    "Strong",    # Sharpen
    "Lens Blur",
    "Lens Blur V2",
    "Motion Blur",
    "Natural",   # Sharpen
    "Refocus",
    # Sharpen - Generative
    "Super Focus V2",
    # Denoise
    "Normal",
    "Strong",    # Denoise
    "Extreme",
    # Restore
    "Dust-Scratch",
    # Lighting
    "Adjust",
    "White Balance"
]

# Format mappings for download accept header
FORMAT_ACCEPT = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff"
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
            },
            "optional": {
                "output_width": ("INT", {
                    "default": 0,  # 0 means auto
                    "min": 1,
                    "max": 32000,
                    "step": 64,
                    "display": "number"
                }),
                "output_height": ("INT", {
                    "default": 0,  # 0 means auto
                    "min": 1,
                    "max": 32000,
                    "step": 64,
                    "display": "number"
                }),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png", "tiff"], {"default": "jpeg"}),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "denoise_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "sharpen_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "fix_compression": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "timeout_seconds": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 1800,
                    "step": 30
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "image/topaz"

    def _get_endpoint_path(self, mode):
        if mode == "restore":
            return "/restore-gen/async"
        elif mode == "enhance":
            return "/enhance/async"
        elif mode == "sharpen":
            return "/sharpen/async"
        elif mode == "denoise":
            return "/denoise/async"
        elif mode == "lighting":
            return "/lighting/async"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _submit_job(self, image_path, api_key, mode, params):
        path = self._get_endpoint_path(mode)
        headers = {"X-API-Key": api_key, "accept": "application/json"}
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}{path}", headers=headers, files=files, data=params)
        response.raise_for_status()
        data = response.json()
        return data["process_id"], data.get("eta", 0)

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{BASE_URL}/getStatus?process_id={process_id}", headers=headers)
            resp.raise_for_status()
            status = resp.json()
            if status["status"] == "completed":
                return True
            if status["status"] == "failed":
                raise Exception(f"Topaz task failed: {status.get('error', 'Unknown')}")
            time.sleep(5)
        raise TimeoutError(f"Task {process_id} timed out after {timeout}s")

    def _download_result(self, process_id, api_key, output_format):
        accept = FORMAT_ACCEPT.get(output_format, "image/jpeg")
        headers = {"X-API-Key": api_key, "accept": accept}
        resp = requests.get(f"{BASE_URL}/getDownloadOutput?process_id={process_id}", headers=headers)
        resp.raise_for_status()
        return resp.content

    def process(self, image, api_key, mode, model,
                output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5, strength=0.5,
                fix_compression=0.0, timeout_seconds=300):

        if not api_key.strip():
            raise ValueError("Topaz API key is required.")

        # Build params dict
        params = {
            "model": model,
            "output_format": output_format,
            "face_enhancement": str(face_enhancement).lower(),
            "denoise_strength": denoise_strength,
            "sharpen_strength": sharpen_strength,
            "strength": strength,
            "fix_compression": fix_compression,
        }

        # Enhance-specific params
        if mode == "enhance":
            if output_width > 0:
                params["output_width"] = output_width
            if output_height > 0:
                params["output_height"] = output_height
            params["crop_to_fill"] = str(crop_to_fill).lower()

        # Convert ComfyUI image to temp file
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        suffix = f".{output_format}" if output_format != "jpeg" else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if output_format == "tiff":
                pil_img.save(tmp.name, format="TIFF")
            elif output_format == "png":
                pil_img.save(tmp.name, format="PNG")
            else:
                pil_img.save(tmp.name, format="JPEG", quality=95)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting {mode} job with model={model}")
            process_id, eta = self._submit_job(input_path, api_key, mode, params)
            print(f"[Topaz] Process ID: {process_id}, ETA: {eta}")

            print(f"[Topaz] Waiting for completion...")
            self._wait_for_completion(process_id, api_key, timeout_seconds)

            print(f"[Topaz] Downloading result...")
            result_bytes = self._download_result(process_id, api_key, output_format)
            result_pil = Image.open(io.BytesIO(result_bytes))

            # Convert to ComfyUI tensor (drop alpha if present)
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]  # RGB only
            result_tensor = result_np[np.newaxis, ...]  # Batch dim

            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
