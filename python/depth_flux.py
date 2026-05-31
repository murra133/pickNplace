"""Wrist image -> local depth map -> Azure FLUX.2-flex (depth-conditioned) -> projector.

No LLM in the loop: the prompt is human-provided (edit PROMPT in project_flux.py
or override with the env var below). Pipeline:
  1. Capture a wrist-cam frame.
  2. Run Depth Anything V2 locally to get a grayscale depth map.
  3. Send depth map + prompt to Azure FLUX.2-flex (depth = control/input image).
  4. Display the result fullscreen on the external monitor.

Reuses capture / Flux / display from project_flux.

Dependencies (local machine):
  pip install transformers pillow torch
  (Flux + display deps come from project_flux.)

Optional env var:
  FLUX_PROMPT   overrides the hardcoded prompt in project_flux.py
"""

import io
import os
import sys
import time

# Use the Rust-based parallel downloader for the multi-GB model weights. Must be
# set before huggingface_hub is imported (it is imported lazily inside the
# generate/depth functions, so setting it here at module load is in time).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import project_flux as pf   # noqa: E402  (also loads .env, incl. HF_TOKEN if set)


# --- Configuration ------------------------------------------------------------
PROMPT       = os.environ.get("FLUX_PROMPT", pf.PROMPT)
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Small-hf"  # swap -Base-/-Large- for quality
DEPTH_OUTPUT = "/Users/brian/Documents/Coding_Projects/pickNplace/depth.png"

# Which generator produces the final image:
#   "gpt_image"  -> Azure gpt-image-2 /images/edits, fed BOTH photo + depth map.
#   "azure_flux" -> Azure FLUX.2-flex, fed the depth map only.
#   "controlnet" -> local SD + depth ControlNet.
GENERATOR = "gpt_image"

# --- gpt-image-2 (Azure OpenAI Images, multi-image edit) ----------------------
# Standard Azure OpenAI resource (https://<res>.openai.azure.com), NOT the BFL
# services.ai.azure.com endpoint used for Flux -- so it has its own env vars.
GPT_IMG_ENDPOINT    = pf._first_env("AZURE_OPENAI_IMAGE_ENDPOINT", "AZURE_OPENAI_ENDPOINT").rstrip("/")
GPT_IMG_KEY         = pf._first_env("AZURE_OPENAI_IMAGE_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
GPT_IMG_DEPLOYMENT  = pf._first_env("AZURE_OPENAI_IMAGE_DEPLOYMENT", default="gpt-image-2")
GPT_IMG_API_VERSION = pf._first_env("AZURE_OPENAI_IMAGE_API_VERSION", default="2025-04-01-preview")
# NOTE: high quality + 1536x1024 + two input images is the heaviest possible
# request and reliably triggers Azure's 408 "operation was timeout". Start light
# and raise once it succeeds. low/1024x1024 is the fastest combo.
# Landscape output to match the 16:9 wrist-cam frame -- avoids squishing a wide
# image into a square. gpt-image sizes: 1024x1024, 1536x1024 (landscape), 1024x1536.
GPT_IMG_SIZE        = "1536x1024"
GPT_IMG_QUALITY     = "low"         # low | medium | high | auto

# --- ControlNet (local, depth-conditioned) -----------------------------------
CN_BASE_MODEL   = "sd-legacy/stable-diffusion-v1-5"          # SD1.5 base (fast on MPS)
CN_CONTROL_MODEL = "lllyasviel/control_v11f1p_sd15_depth"    # depth ControlNet
# How strictly the output follows the depth map. 1.0 = strong; raise toward 1.5
# to lock geometry harder, lower toward 0.6 to let the prompt roam more.
CN_CONDITIONING_SCALE = 1.0
# img2img denoising strength: how far to transform the camera photo.
#   low  (0.3-0.5) -> output stays close to the real photo
#   high (0.7-0.9) -> output becomes a full rendering, photo only seeds composition
CN_STRENGTH      = 0.8
CN_STEPS         = 28
CN_GUIDANCE      = 7.0
CN_NEGATIVE      = "blurry, distorted, low quality, watermark, text"
CN_WIDTH         = 768    # multiple of 8; 768x432 = 16:9
CN_HEIGHT        = 432
CN_SEED          = None   # int for reproducible output, None for random-each-run

# --- Depth normalization knobs ------------------------------------------------
# These shape the grayscale. The goal: spend the full 0-255 range on the band of
# heights you care about, and amplify small differences between nearby surfaces.

# Polarity. Depth Anything outputs inverse depth (near = large). If "high" spots
# come out dark when they should be light (or vice versa), flip this.
DEPTH_INVERT = False

# Depth window, as percentiles of the depth distribution. Only this band is
# stretched across the gray ramp; anything FARTHER than the low edge collapses
# to flat (i.e. ignored). To "focus on near surfaces and ignore far away,"
# raise the low number. (0,100) = old behavior (use everything).
#   e.g. (60, 100) keeps the nearest ~40% of the depth range at full contrast.
#   (0, 100) = no range filtering; show the full depth as-is.
DEPTH_WINDOW_PCT = (0, 100)

# Local contrast: 0 = off. >0 mixes in a high-pass that amplifies differences
# between NEIGHBORING surfaces regardless of their absolute depth -- this is the
# "two things at similar height, show their difference" control. Try 0.5-0.8.
DEPTH_LOCAL_CONTRAST = 0.6
DEPTH_LOCAL_RADIUS   = 30      # px; neighborhood scale for the high-pass
DEPTH_LOCAL_GAIN     = 4.0     # how hard to amplify local differences

# Tone shaping after windowing.
DEPTH_GAMMA      = 1.0         # <1 brightens mids, >1 darkens mids
DEPTH_GRAY_RANGE = (0, 255)    # output window; e.g. (30, 235) to avoid pure black/white

_depth_pipe = None


def _normalize_depth(raw):
    """raw: 2D float numpy array of predicted depth -> uint8 grayscale array."""
    import numpy as np
    from PIL import Image, ImageFilter

    v = raw.astype("float32")
    if DEPTH_INVERT:
        v = -v

    # Window: stretch the chosen percentile band across [0,1]; clip the rest.
    lo = np.percentile(v, DEPTH_WINDOW_PCT[0])
    hi = np.percentile(v, DEPTH_WINDOW_PCT[1])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    n = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    if DEPTH_GAMMA != 1.0:
        n = np.power(n, DEPTH_GAMMA)

    # Local contrast: high-pass = detail above a blurred baseline. Centers on 0.5
    # so flat regions stay mid-gray and only neighbor differences pop.
    if DEPTH_LOCAL_CONTRAST > 0:
        base_img = Image.fromarray((n * 255).astype("uint8"), mode="L")
        blurred = np.asarray(
            base_img.filter(ImageFilter.GaussianBlur(radius=DEPTH_LOCAL_RADIUS)),
            dtype="float32",
        ) / 255.0
        high_pass = np.clip(0.5 + (n - blurred) * DEPTH_LOCAL_GAIN, 0.0, 1.0)
        n = (1.0 - DEPTH_LOCAL_CONTRAST) * n + DEPTH_LOCAL_CONTRAST * high_pass

    g_lo, g_hi = DEPTH_GRAY_RANGE
    gray = g_lo + n * (g_hi - g_lo)
    return np.clip(gray, 0, 255).astype("uint8")


# --- Local depth estimation ---------------------------------------------------
def compute_depth_map(jpg_bytes: bytes) -> bytes:
    """Run Depth Anything V2 locally; return a tuned grayscale depth PNG."""
    global _depth_pipe
    import numpy as np
    from PIL import Image

    if _depth_pipe is None:
        from transformers import pipeline
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
        print(f"  loading {DEPTH_MODEL} on {device} (first run downloads weights)...")
        _depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=device)

    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    out = _depth_pipe(img)

    # Use the raw, full-precision tensor (not out["depth"], which is pre-normalized).
    raw = out["predicted_depth"]
    raw = raw.squeeze().detach().cpu().numpy()
    gray = _normalize_depth(raw)

    depth_img = Image.fromarray(gray, mode="L").resize(img.size)
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    png = buf.getvalue()
    with open(DEPTH_OUTPUT, "wb") as f:
        f.write(png)
    w, h = depth_img.size
    print(f"  depth map computed ({w}x{h}) window={DEPTH_WINDOW_PCT} "
          f"local={DEPTH_LOCAL_CONTRAST}, saved {DEPTH_OUTPUT}")
    return png


# --- ControlNet generation (local, depth-conditioned) ------------------------
_cn_pipe = None


def generate_controlnet(photo_jpg: bytes, depth_png: bytes, prompt: str) -> bytes:
    """img2img on the camera photo, with depth ControlNet locking the geometry."""
    global _cn_pipe
    import torch
    from PIL import Image
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if _cn_pipe is None:
        print(f"  loading ControlNet img2img pipeline on {device} (first run downloads ~5GB)...")
        controlnet = ControlNetModel.from_pretrained(CN_CONTROL_MODEL, torch_dtype=torch.float32)
        _cn_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            CN_BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)

    init_img    = Image.open(io.BytesIO(photo_jpg)).convert("RGB").resize((CN_WIDTH, CN_HEIGHT))
    control_img = Image.open(io.BytesIO(depth_png)).convert("RGB").resize((CN_WIDTH, CN_HEIGHT))

    generator = None
    if CN_SEED is not None:
        generator = torch.Generator(device=device).manual_seed(CN_SEED)

    result = _cn_pipe(
        prompt=prompt,
        image=init_img,                 # img2img seed = the camera photo
        control_image=control_img,      # ControlNet conditioning = the depth map
        strength=CN_STRENGTH,
        negative_prompt=CN_NEGATIVE,
        num_inference_steps=CN_STEPS,
        guidance_scale=CN_GUIDANCE,
        controlnet_conditioning_scale=CN_CONDITIONING_SCALE,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# --- gpt-image-2 generation (Azure OpenAI, both images) ----------------------
def _multipart(fields: dict, files: list) -> tuple:
    """Build multipart/form-data. files = [(field, filename, mime, bytes), ...].
    Returns (content_type, body_bytes)."""
    import uuid
    boundary = "----pickNplace-" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts = []
    for name, value in fields.items():
        parts += [b"--" + boundary.encode(),
                  f'Content-Disposition: form-data; name="{name}"'.encode(),
                  b"", str(value).encode()]
    for field, filename, mime, data in files:
        parts += [b"--" + boundary.encode(),
                  f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode(),
                  f"Content-Type: {mime}".encode(),
                  b"", data]
    parts += [b"--" + boundary.encode() + b"--", b""]
    return f"multipart/form-data; boundary={boundary}", crlf.join(parts)


def generate_gpt_image(photo_jpg: bytes, depth_png: bytes, prompt: str) -> bytes:
    """Azure gpt-image-2 /images/edits with BOTH the photo and the depth map."""
    import json
    import urllib.error
    import urllib.request

    # Thorough projection-mapping instruction. The output is projected back ON TOP
    # of the real physical scene, so it must register with Image 1 pixel-for-pixel.
    full_prompt = (
        "TASK: projection-mapping overlay. The image you produce will be projected "
        "directly back on top of the exact physical scene shown in Image 1, so it "
        "MUST line up with Image 1 pixel-for-pixel.\n\n"
        "INPUTS: Two images of the same scene from the same camera. "
        "Image 1 is the real photo: boxes placed on a white grid surface. "
        "Image 2 is a depth map of that scene where BRIGHTER = physically HIGHER / "
        "closer and darker = lower.\n\n"
        "HARD RULES:\n"
        "1. DO NOT change the camera view, framing, perspective, scale, zoom, "
        "rotation, or crop. Keep the identical viewpoint and composition as Image 1. "
        "The output must overlay Image 1 exactly.\n"
        "2. Keep the scene visually UNCHANGED everywhere except on the raised boxes. "
        "The white grid, the table/background, lighting, colors, and the position of "
        "everything must stay exactly as in Image 1.\n"
        "3. Use Image 2 to find the raised areas. ONLY where a box is clearly raised "
        "above the grid, build one small building/home on that exact footprint, with "
        "the raised top as its roof.\n"
        "4. Where the grid is flat with no raised box, add NOTHING -- leave it as the "
        "bare grid.\n"
        "5. Do not invent, move, duplicate, resize, or remove anything. Same number "
        "and positions of structures as there are raised boxes.\n"
        "6. Output the same resolution and aspect ratio as Image 1.\n\n"
        "STYLE: " + prompt
    )
    fields = {
        "model":   GPT_IMG_DEPLOYMENT,
        "prompt":  full_prompt,
        "size":    GPT_IMG_SIZE,
        "quality": GPT_IMG_QUALITY,
        "n":       "1",
    }
    files = [
        ("image[]", "scene.jpg", "image/jpeg", photo_jpg),
        ("image[]", "depth.png", "image/png",  depth_png),
    ]
    content_type, body = _multipart(fields, files)
    # Two Azure URL shapes:
    #   new "v1" surface:  https://<res>.cognitiveservices.azure.com/openai/v1
    #        -> {endpoint}/images/edits   (model in body, no api-version query)
    #   classic surface:   https://<res>.openai.azure.com
    #        -> {endpoint}/openai/deployments/{deployment}/images/edits?api-version=...
    if "/openai/v1" in GPT_IMG_ENDPOINT:
        url = f"{GPT_IMG_ENDPOINT}/images/edits"
    else:
        url = (f"{GPT_IMG_ENDPOINT}/openai/deployments/{GPT_IMG_DEPLOYMENT}"
               f"/images/edits?api-version={GPT_IMG_API_VERSION}")
    req = urllib.request.Request(
        url, method="POST", data=body,
        headers={"api-key": GPT_IMG_KEY, "Content-Type": content_type, "Accept": "application/json"},
    )
    print(f"  POST {url}")
    try:
        with urllib.request.urlopen(req, timeout=180, context=pf._SSL_CTX) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"gpt-image HTTP {e.code}: {e.read().decode('utf-8','replace')[:500]}") from None

    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"no data in gpt-image response: {str(payload)[:400]}")
    item = data[0]
    if item.get("b64_json"):
        import base64
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=60, context=pf._SSL_CTX) as r:
            return r.read()
    raise RuntimeError(f"no image bytes in gpt-image result: {item}")


def generate(photo_jpg: bytes, depth_png: bytes, prompt: str) -> bytes:
    if GENERATOR == "gpt_image":
        return generate_gpt_image(photo_jpg, depth_png, prompt)
    if GENERATOR == "controlnet":
        return generate_controlnet(photo_jpg, depth_png, prompt)
    return pf.run_flux(depth_png, prompt)   # azure_flux fallback (depth only)


# --- Main ---------------------------------------------------------------------
def main():
    if not pf.ROBOT_TOKEN:
        sys.exit("ROBOT_TOKEN not set in project_flux.py")
    if GENERATOR == "azure_flux" and (not pf.AZURE_ENDPOINT or not pf.AZURE_KEY):
        sys.exit("Flux endpoint/key not set (AZURE_OPENAI_* env vars)")
    if GENERATOR == "gpt_image" and (not GPT_IMG_ENDPOINT or not GPT_IMG_KEY):
        sys.exit("gpt-image endpoint/key not set (AZURE_OPENAI_IMAGE_* env vars)")

    print("capturing wrist-cam frame...")
    jpg = pf.capture_wrist_frame()
    with open(pf.SOURCE_SNAPSHOT, "wb") as f:
        f.write(jpg)
    print(f"  saved {pf.SOURCE_SNAPSHOT} ({len(jpg)} bytes)")

    print("computing depth map...")
    depth_png = compute_depth_map(jpg)

    print(f"generating image via '{GENERATOR}'...")
    t0 = time.time()
    result = generate(jpg, depth_png, PROMPT)
    print(f"  got generated image ({len(result)} bytes) in {time.time() - t0:.1f}s")
    with open(pf.OUTPUT_PATH, "wb") as f:
        f.write(result)
    print(f"  saved {pf.OUTPUT_PATH}")

    pf.show_fullscreen(result)
    print("  displayed on external monitor. Ctrl-C to exit.")
    try:
        while True:
            pf.pump_cocoa_events(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
