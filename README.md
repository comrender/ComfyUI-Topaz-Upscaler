# ComfyUI-Topaz-Upscaler
A custom node for ComfyUI that integrates the Topaz Labs API for AI-powered image upscaling and enhancement.

#Features

Upscale images using Topaz Labs' advanced AI models.
Configurable output width (512 to 16000 pixels).
Option to crop-to-fill for aspect ratio adjustment.
API key input directly in the node UI (no hardcoding).
Supports JPEG input/output with automatic tensor conversion.
Error handling, timeouts, and temporary file cleanup.

#Requirements

ComfyUI installed.
A valid Topaz Labs API key (sign up at topazlabs.com).
Python libraries: requests, Pillow, numpy (usually pre-installed in ComfyUI environments).

Note: The free tier of Topaz API has usage limits; consider a paid plan for heavy use.

#Installation

Navigate to your ComfyUI custom nodes directory:textcd ComfyUI/custom_nodes
Clone or create the folder:textmkdir ComfyUI-Topaz-Upscaler
cd ComfyUI-Topaz-Upscaler
Add the following files:
topaz_upscaler.py
__init__.py

Restart ComfyUI or use the "Reload custom nodes" option in the manager.
Search for "Topaz Upscaler" in the ComfyUI node search menu.

#Usage

Add the Topaz Upscaler node to your workflow.
Connect an image input (from Load Image or other nodes).
Enter your Topaz API key in the node's "api_key" field.
Set the desired output width and crop_to_fill option.
Run the workflow. The node will submit the job to Topaz API, wait for completion, and output the upscaled image tensor.

#Example Workflow

Input: Low-res image → Topaz Upscaler → Preview Image or Save Image.

#Troubleshooting

API Key Error: Ensure your key is valid and has credits.
Timeout: Increase the timeout in code if jobs take longer (default: 300s).
No Output: Check console for logs like "[Topaz Upscaler] Task ID: ...".
If issues persist, verify Topaz API status at topazlabs.com.

#Improvements & Notes

Merged API and node logic into a single file for simplicity.
Proper temp file handling to avoid leaks.
Output is JPEG from API, converted to ComfyUI-compatible tensor.
For PNG support or batch processing, open an issue or contribute!

#License
MIT License. Feel free to use, modify, and distribute.
#Credits
Based on original code snippets for Topaz API integration. Enhanced for ComfyUI compatibility.
