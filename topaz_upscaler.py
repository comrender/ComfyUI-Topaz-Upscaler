import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io
import math

BASE_URL = "https://api.topazlabs.com/image/v1"

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
                "output_width": ("INT", {
                    "default": 3840,
                    "min": 512,
                    "max": 16000,
                    "step": 64
                }),
            },
            "optional": {
                "scale_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {
                    "default": 180,  # Matches original
                    "min": 30,
                    "max": 1800,
                    "step": 30
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def _submit_job(self, image_path, api_key, output_width, crop_to_fill):
        headers = {
            "X-API-Key": api_key,
            "accept": "application/json"
        }
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                "output_width": output_width,
                "crop_to_fill": str(crop_to_fill).lower(),
                "output_format": "jpeg"
            }
            response = requests.post(f"{BASE_URL}/enhanceAsync", headers=headers, files=files, data=data)
        response.raise_for_status()
        data = response.json()
        print(f"[Topaz] Submit response: {data}")
        
        # Extract task_id (original format)
        task_id = data.get("task_id")
        if not task_id:
            # Fallback to process_id or header if API changed
            task_id = data.get("process_id") or response.headers.get("X-Process-ID")
            if not task_id:
                raise ValueError(f"No task_id/process_id in response: {data}")
        
        # Check for ETA header (Unix timestamp)
        eta_header = response.headers.get("X-ETA")
        eta = int(eta_header) if eta_header else None
        if eta:
            remaining = max(eta - int(time.time()), 0)
            print(f"[Topaz] ETA remaining: ~{remaining}s (adjusted timeout)")
        
        return task_id, eta

    def _wait_for_completion(self, task_id, api_key, timeout):
        headers = {"X-API-Key": api_key}
        start_time = time.time()
        poll_interval = 5  # Matches original
        
        status_url = f"{BASE_URL}/getStatus?task_id={task_id}"
        print(f"[Topaz] Status URL: {status_url}")
        
        while time.time() - start_time < timeout:
            resp = requests.get(status_url, headers=headers)
            print(f"[Topaz] Status code: {resp.status_code}")
            
            if resp.status_code == 404:
                print("[Topaz] 404 on status - job may be pending or API mismatch. Retrying...")
                time.sleep(poll_interval)
                continue
            
            resp.raise_for_status()
            status = resp.json()
            print(f"[Topaz] Status: {status.get('status', 'unknown')}")
            
            if status.get("status") == "completed":
                return True
            if status.get("status") == "failed":
                raise Exception(f"Job failed: {status.get('error', 'Unknown')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {task_id} timed out after {timeout}s. Check Topaz dashboard for manual download.")

    def _download_result(self, task_id, api_key):
        headers = {
            "X-API-Key": api_key,
            "accept": "image/jpeg"
        }
        download_url = f"{BASE_URL}/getDownloadOutput?task_id={task_id}"
        print(f"[Topaz] Download URL: {download_url}")
        
        resp = requests.get(download_url, headers=headers)
        print(f"[Topaz] Download code: {resp.status_code}")
        
        if resp.status_code == 404:
            raise Exception(f"Download 404 - ensure job completed. Manual check: {download_url}")
        
        resp.raise_for_status()
        return resp.content

    def upscale(self, image, api_key, output_width, scale_multiplier=1.0, crop_to_fill=False, timeout_seconds=180):
        if not api_key.strip():
            raise ValueError("Topaz API key is required.")

        # === AUTO-SCALE ===
        h, w = image.shape[1], image.shape[2]
        if scale_multiplier != 1.0:
            new_w = int(round(w * scale_multiplier))
            output_width = min(max(new_w, 512), 16000)
            print(f"[Topaz] Scale ×{scale_multiplier} → {w}×{h} → width={output_width}")

        # === TEMP INPUT ===
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp.name, format="JPEG", quality=95)
            input_path = tmp.name

        try:
            print(f"[Topaz] Submitting job | Width: {output_width}, Crop: {crop_to_fill}")
            task_id, eta = self._submit_job(input_path, api_key, output_width, crop_to_fill)

            print(f"[Topaz] Waiting for completion...")
            self._wait_for_completion(task_id, api_key, timeout_seconds)

            print(f"[Topaz] Downloading...")
            result_bytes = self._download_result(task_id, api_key)

            # FIXED: Load from BytesIO (original bug)
            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                result_np = result_np[:, :, :3]  # Drop alpha if PNG
            result_tensor = result_np[np.newaxis, ...]  # Batch dim

            print(f"[Topaz] Success! Output shape: {result_tensor.shape}")
            return (result_tensor,)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


NODE_CLASS_MAPPINGS = {"TopazUpscaler": TopazUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TopazUpscaler": "Topaz Upscaler"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
