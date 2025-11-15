import os
import time
import requests
import tempfile
import numpy as np
from PIL import Image
import io

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
                "crop_to_fill": ("BOOLEAN", {"default": False}),
            },
            "optional": {}
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
        return response.json()["task_id"]

    def _wait_for_completion(self, task_id, api_key, timeout=300):
        headers = {"X-API-Key": api_key}
        start_time = time.time()
        while time.time() - start_time < timeout:
            resp = requests.get(f"{BASE_URL}/getStatus?task_id={task_id}", headers=headers)
            resp.raise_for_status()
            status = resp.json()
            if status["status"] == "completed":
                return True
            elif status["status"] == "failed":
                raise Exception(f"Topaz task {task_id} failed: {status.get('error', 'Unknown error')}")
            time.sleep(5)
        raise TimeoutError(f"Topaz task {task_id} timed out after {timeout} seconds.")

    def _download_result(self, task_id, api_key):
        headers = {
            "X-API-Key": api_key,
            "accept": "image/jpeg"
        }
        resp = requests.get(f"{BASE_URL}/getDownloadOutput?task_id={task_id}", headers=headers)
        resp.raise_for_status()
        return resp.content

    def upscale(self, image, api_key, output_width, crop_to_fill):
        if not api_key or api_key.strip() == "":
            raise ValueError("Topaz API key is required. Please enter it in the node.")

        # Convert ComfyUI image tensor [B,H,W,C] float32 [0,1] -> PIL Image
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            pil_image.save(tmp_input.name, format="JPEG", quality=95)
            input_path = tmp_input.name

        try:
            print(f"[Topaz Upscaler] Submitting job with width={output_width}, crop={crop_to_fill}")
            task_id = self._submit_job(input_path, api_key, output_width, crop_to_fill)
            print(f"[Topaz Upscaler] Task ID: {task_id}")

            print(f"[Topaz Upscaler] Waiting for completion...")
            self._wait_for_completion(task_id, api_key)

            print(f"[Topaz Upscaler] Downloading result...")
            result_bytes = self._download_result(task_id, api_key)

            # Load result
            result_pil = Image.open(io.BytesIO(result_bytes))
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            result_tensor = result_np[np.newaxis, ...]  # Add batch dim

            return (result_tensor,)

        finally:
            # Clean up input temp file
            if os.path.exists(input_path):
                os.unlink(input_path)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TopazUpscaler": TopazUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazUpscaler": "Topaz Upscaler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']