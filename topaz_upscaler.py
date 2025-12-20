import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io
import torch

BASE_URL = "https://api.topazlabs.com/image/v1"

# Updated Model Category Mappings based on Topaz API Docs
ENHANCE_GAN_MODELS = ["Standard V2", "Low Resolution V2", "CGI", "High Fidelity V2", "Text Refine"]
ENHANCE_GEN_MODELS = ["Redefine", "Recovery V2", "Standard MAX", "Wonder"]
SHARPEN_GAN_MODELS = ["Standard", "Strong", "Lens Blur", "Lens Blur V2", "Motion Blur", "Natural", "Refocus"]
SHARPEN_GEN_MODELS = ["Super Focus V2"]
DENOISE_GAN_MODELS = ["Normal", "Strong", "Extreme"]
RESTORE_GEN_MODELS = ["Dust-Scratch"]
LIGHTING_GAN_MODELS = ["Adjust", "White Balance"]

TOPAZ_MODES = ["enhance", "sharpen", "denoise", "restore", "lighting"]
TOPAZ_MODELS = list(set(
    ENHANCE_GAN_MODELS + ENHANCE_GEN_MODELS +
    SHARPEN_GAN_MODELS + SHARPEN_GEN_MODELS +
    DENOISE_GAN_MODELS + RESTORE_GEN_MODELS + LIGHTING_GAN_MODELS
))

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
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Enter Topaz API Key"}),
                "mode": (TOPAZ_MODES, {"default": "enhance"}),
                "model": (TOPAZ_MODELS, {"default": "Standard V2"}),
                "scale_multiplier": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "output_width": ("INT", {"default": 0, "min": 0, "max": 32000}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 32000}),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png", "tiff"], {"default": "jpeg"}),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "sharpen_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "timeout_seconds": ("INT", {"default": 300, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/topaz"

    def _get_submit_path(self, mode, model):
        mapping = {
            "enhance": ("/enhance-gen/async" if model in ENHANCE_GEN_MODELS else "/enhance/async"),
            "sharpen": ("/sharpen-gen/async" if model in SHARPEN_GEN_MODELS else "/sharpen/async"),
            "denoise": "/denoise/async",
            "restore": "/restore-gen/async",
            "lighting": "/lighting/async"
        }
        if mode in mapping:
            return mapping[mode]
        raise ValueError(f"Invalid mode/model combo: {mode}/{model}")

    def _submit_job(self, image_path, api_key, mode, model, params):
        path = self._get_submit_path(mode, model)
        headers = {"X-API-Key": api_key}
        
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f)}
            # API expects multipart form-data for async endpoints
            data = {k: str(v) for k, v in params.items()}
            response = requests.post(f"{BASE_URL}{path}", headers=headers, files=files, data=data, timeout=60)
        
        response.raise_for_status()
        resp_json = response.json()
        process_id = resp_json.get("process_id")
        if not process_id:
            raise ValueError(f"No process_id in response: {resp_json}")
        return process_id

    def _wait_for_completion(self, process_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start = time.time()
        url = f"{BASE_URL}/status/{process_id}"

        while time.time() - start < timeout:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 404:
                time.sleep(5); continue
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "").lower()
            
            if status == "completed":
                return True
            if status == "failed":
                raise Exception(f"Topaz Job Failed: {data.get('error', 'Unknown Error')}")
            
            time.sleep(5)
        raise TimeoutError(f"Job {process_id} timed out.")

    def _download_result(self, process_id, api_key, output_format):
        headers = {"X-API-Key": api_key}
        url = f"{BASE_URL}/download/{process_id}"
        
        # Get the presigned URL
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        dl_url = resp.json().get("download_url")
        
        # Download the binary
        img_resp = requests.get(dl_url, timeout=60)
        img_resp.raise_for_status()
        return img_resp.content

    def process(self, image, api_key="", mode="enhance", model="Standard V2",
                scale_multiplier=1.0, output_width=0, output_height=0, crop_to_fill=False,
                output_format="jpeg", face_enhancement=True,
                denoise_strength=0.5, sharpen_strength=0.5, strength=0.5,
                timeout_seconds=300):

        if not api_key.strip():
            api_key = os.getenv("TOPAZ_API_KEY", "")
            if not api_key: raise ValueError("API Key Missing")

        # image is [B, H, W, C]
        _, h, w, _ = image.shape

        if scale_multiplier > 1.0:
            output_width = int(w * scale_multiplier)
            output_height = int(h * scale_multiplier)

        params = {
            "model": model,
            "output_format": output_format,
            "face_enhancement": str(face_enhancement).lower(),
            "denoise_strength": denoise_strength,
            "sharpen_strength": sharpen_strength,
            "strength": strength,
        }
        if mode == "enhance":
            if output_width > 0: params["output_width"] = output_width
            if output_height > 0: params["output_height"] = output_height
            params["crop_to_fill"] = str(crop_to_fill).lower()

        # Save ComfyUI tensor to PIL -> Temp File
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            pil_img.save(tmp.name, format=output_format.upper())
            input_path = tmp.name

        try:
            pid = self._submit_job(input_path, api_key, mode, model, params)
            self._wait_for_completion(pid, api_key, timeout_seconds)
            bytes_data = self._download_result(pid, api_key, output_format)

            # Convert back to ComfyUI Tensor
            result_pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            return (torch.from_numpy(result_np).unsqueeze(0),)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler (API)"}
