"""Wrist-cam -> Azure-hosted FLUX.2-flex -> fullscreen on HDMI projector.

Pipeline (one shot, or looped if LOOP_SECONDS > 0):
  1. Pull a single JPEG from the RO1's MJPEG wrist-cam stream.
  2. POST it + PROMPT to FLUX.2-flex via Azure's OpenAI-compatible Images API.
  3. Decode the inline base64 PNG and show it fullscreen on the second monitor.

Dependencies:  pip install pyobjc-framework-Cocoa
(All HTTP is stdlib; display is native Cocoa via pyobjc.)

FLUX.2-flex on Azure AI Foundry exposes the OpenAI-compatible Images API
(synchronous /images/edits), so this is just one HTTP call with a multipart
body and the result comes back inline as base64 -- no polling, no BFL job IDs.

Env vars read (first non-empty wins; common Azure naming conventions covered):
  AZURE_OPENAI_ENDPOINT | AZURE_FLUX_ENDPOINT       base URL of the resource
  AZURE_OPENAI_API_KEY  | AZURE_OPENAI_KEY |
                          AZURE_FLUX_KEY            api key
  AZURE_OPENAI_DEPLOYMENT | AZURE_FLUX_DEPLOYMENT   deployment name (defaults
                                                    to AZURE_MODEL constant below)
  AZURE_OPENAI_API_VERSION                          api-version query param
                                                    (default 2025-04-01-preview)
"""

import base64
import io
import json
import os
import pathlib
import select
import ssl
import sys
import termios
import time
import tty
import urllib.error
import urllib.request

# Display is native Cocoa (AppKit/Foundation via pyobjc) -- no OpenCV needed.


# --- .env loader (stdlib only) ------------------------------------------------
def _load_dotenv() -> None:
    """Read a .env file and set any vars that aren't already in the environment.

    Looks for the file in (1) the script directory, (2) the repo root one level
    up, (3) the current working directory. Shell env always wins over .env.
    """
    here = pathlib.Path(__file__).resolve().parent
    candidates = [here / ".env", here.parent / ".env", pathlib.Path.cwd() / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        return  # only load the first .env found


_load_dotenv()


# --- Configuration ------------------------------------------------------------
ROBOT_URL      = os.environ.get("ROBOT_URL", "https://cb2347.sb.app")
ROBOT_TOKEN    = os.environ.get("ROBOT_TOKEN", "")   # set in .env


def _first_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# For the Azure AI Foundry BFL provider, this is the FULL submit URL --
# including the `?api-version=preview` query string -- not just a host.
# Example:
#   https://<resource>.services.ai.azure.com/providers/blackforestlabs/v1/flux-2-flex?api-version=preview
AZURE_ENDPOINT = _first_env("AZURE_OPENAI_ENDPOINT", "AZURE_FLUX_ENDPOINT")
AZURE_KEY      = _first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_FLUX_KEY")
AZURE_MODEL    = "FLUX.2-flex"   # logged in output only; the URL already names the model

# Hardcoded prompt -- edit this freely.
# Only the STYLE half of the instruction; the registration/fidelity rules live in
# depth_flux's wrapper. Each raised box becomes a home; its raised top is the roof.
PROMPT = (
    "Render each raised box as a realistic small home/building sitting on its grid "
    "plot, with the raised top as the roof. Keep it photorealistic and faithful, "
    "matching the existing footprint and position exactly."
)

# Output aspect ratio for the projector.
ASPECT_RATIO = "16:9"

# Output / display.
# Leave PROJECTOR_X/Y as None to auto-detect the external monitor's origin via
# AppKit. Set explicit integers to override (top-left corner in logical pixels).
PROJECTOR_X        = None
PROJECTOR_Y        = None
OUTPUT_PATH        = "/Users/brian/Documents/Coding_Projects/pickNplace/generated.png"
SOURCE_SNAPSHOT    = "/Users/brian/Documents/Coding_Projects/pickNplace/snapshot.jpg"
LOOP_SECONDS       = 0   # 0 = one shot. e.g. 30 = regenerate every 30s.

# Bypass cert verification (matches what we did against the robot's https endpoint).
_SSL_CTX = ssl._create_unverified_context()

# Software darkening applied to EVERY captured frame (the wrist cam is
# over-exposed). Applied centrally here so all consumers -- camera_feed,
# depth_flux, project_flux -- get the same corrected image.
#   DARKEN_GAIN   overall brightness multiplier (1.0 = no change, lower = darker)
#   DARKEN_GAMMA  >1 pulls highlights down harder than shadows (tames glare)
DARKEN_GAIN  = 0.75
DARKEN_GAMMA = 1.2


def _darken(jpg_bytes: bytes) -> bytes:
    """Darken a JPEG via gain + gamma using a per-channel LUT. Returns JPEG bytes."""
    if DARKEN_GAIN == 1.0 and DARKEN_GAMMA == 1.0:
        return jpg_bytes
    from PIL import Image
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    lut = []
    for i in range(256):
        v = DARKEN_GAIN * (i / 255.0)
        v = max(0.0, min(1.0, v)) ** DARKEN_GAMMA
        lut.append(int(round(v * 255)))
    img = img.point(lut * 3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# --- Camera capture -----------------------------------------------------------
# "frame"  -> Option A: GET /api/v1/camera/frame/rgb with a camera_settings body.
# "stream" -> Option C: scan the MJPEG stream for one JPEG.
CAPTURE_METHOD = "frame"

# Sent as the camera_settings body for the frame/rgb endpoint (required -- a null
# body returns 400 "Missing camera settings").
CAMERA_SETTINGS = {
    "brightness": 50,
    "contrast": 50,
    "exposure": 100,
    "sharpness": 50,
    "hue": 0,
    "whiteBalance": 4000,
    "autoWhiteBalance": True,
}

_CAM_HEADERS = {
    "Authorization": "Bearer " + ROBOT_TOKEN,
    "robot-kind": "live",
    "robot_kind": "live",
}


def _capture_stream() -> bytes:
    """Option C: pull one JPEG out of the MJPEG stream."""
    req = urllib.request.Request(ROBOT_URL + "/api/v1/camera/stream/rgb", headers=_CAM_HEADERS)
    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
        chunk = r.read(1_048_576)
    soi = chunk.find(b"\xff\xd8")
    eoi = chunk.find(b"\xff\xd9", soi + 2) if soi != -1 else -1
    if soi == -1 or eoi == -1:
        raise RuntimeError("no JPEG markers found in MJPEG stream")
    return chunk[soi:eoi + 2]


def _decode_frame_response(data: bytes) -> bytes:
    """frame/rgb returns a `data:image/jpeg;base64,...` URI; also handle raw/JSON."""
    if data[:2] == b"\xff\xd8":                       # raw JPEG
        return data
    text = data.decode("utf-8", "ignore").strip()
    if text.startswith("data:"):                      # data URI -> take part after comma
        return base64.b64decode(text.split(",", 1)[1])
    try:                                              # JSON envelope
        obj = json.loads(data)
        b64 = (obj.get("imageData") or obj.get("image")
               or obj.get("data") or obj.get("b64_json") or "")
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass
    try:                                              # plain base64 text
        return base64.b64decode(text)
    except Exception:
        raise RuntimeError(f"could not decode frame/rgb response: {data[:120]!r}")


def _capture_frame_rgb() -> bytes:
    """Option A: GET /api/v1/camera/frame/rgb with a camera_settings JSON body."""
    body = json.dumps({"camera_settings": CAMERA_SETTINGS}).encode()
    req = urllib.request.Request(
        ROBOT_URL + "/api/v1/camera/frame/rgb",
        data=body, method="GET",
        headers={**_CAM_HEADERS, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
        return _decode_frame_response(r.read())


def capture_wrist_frame() -> bytes:
    """Capture one wrist-cam JPEG (per CAPTURE_METHOD), darkened. Returns JPEG bytes."""
    raw = _capture_frame_rgb() if CAPTURE_METHOD == "frame" else _capture_stream()
    return _darken(raw)


def set_camera_settings(**settings) -> tuple:
    """Set wrist-camera settings (e.g. exposure, brightness, contrast).

    Hits the same endpoint the Standard Bots web UI uses. Settings persist on the
    camera, so this only needs to be called once before capturing. Valid keys
    (ints unless noted): brightness, contrast, exposure, sharpness, hue,
    whiteBalance, autoWhiteBalance (bool). Lower `exposure` = darker image.
    Returns (status_code, response_body).
    """
    body = {"kind": "wristCamera", **settings}
    req = urllib.request.Request(
        ROBOT_URL + "/camera-bot-api/camera/settings",
        method="POST",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + ROBOT_TOKEN,
            "robot-kind": "live",
            "robot_kind": "live",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
            return r.status, r.read().decode("utf-8", "replace")[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]


# --- Azure AI Foundry (BFL provider) -----------------------------------------
# POST the endpoint URL exactly as configured (it already includes the
# `?api-version=preview` query and the model name in the path). Body is JSON
# in BFL's native schema. Response can be synchronous (image inline) OR
# async (polling_url) depending on the model and load; handle both.
def _bfl_get(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"api-key": AZURE_KEY, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
        return json.loads(r.read())


def _extract_image(payload: dict) -> bytes:
    """Pull image bytes out of whichever response shape Azure returned."""
    # 1) BFL: result.sample = URL  (after polling)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else None
    sample = (
        (result or {}).get("sample")
        or payload.get("sample")
        or payload.get("image_url")
        or payload.get("url")
    )
    # 2) Some shapes nest under "image": {"url": "..."}.
    if not sample and isinstance(payload.get("image"), dict):
        sample = payload["image"].get("url") or payload["image"].get("image_url")
    if sample:
        if sample.startswith("data:"):
            return base64.b64decode(sample.split(",", 1)[1])
        with urllib.request.urlopen(sample, timeout=60, context=_SSL_CTX) as r:
            return r.read()
    # 3) Base64 inline.
    b64 = payload.get("b64_json")
    if not b64:
        data = payload.get("data")
        if isinstance(data, list) and data:
            item = data[0]
            b64 = item.get("b64_json") if isinstance(item, dict) else None
    if b64:
        return base64.b64decode(b64)
    raise RuntimeError(f"no image in response: {str(payload)[:500]}")


def run_flux(jpg_bytes: bytes, prompt: str) -> bytes:
    """POST one image+prompt to the BFL provider on Azure; return generated bytes."""
    body = {
        "model":           AZURE_MODEL,
        "prompt":          prompt,
        "input_image":     base64.b64encode(jpg_bytes).decode(),
        "aspect_ratio":    ASPECT_RATIO,
        "output_format":   "png",
        "safety_tolerance": 2,
    }
    req = urllib.request.Request(
        AZURE_ENDPOINT,
        method="POST",
        headers={
            "api-key":      AZURE_KEY,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        },
        data=json.dumps(body).encode(),
    )
    print(f"  POST {AZURE_ENDPOINT}")
    try:
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure HTTP {e.code}: {err_body[:500]}") from None

    # Async path: poll until Ready.
    poll = payload.get("polling_url") or payload.get("polling_uri")
    if poll:
        deadline = time.time() + 180
        while True:
            if time.time() > deadline:
                raise RuntimeError("Azure Flux polling timed out after 180s")
            time.sleep(1.0)
            result = _bfl_get(poll)
            status = result.get("status", "")
            if status in ("Ready", "succeeded", "Succeeded"):
                payload = result
                break
            if status in ("Failed", "Error", "Content Moderated", "Request Moderated"):
                raise RuntimeError(f"Azure Flux prediction failed: {status}: {result}")
            # Pending / Processing / Task not found (transient) -> keep polling

    return _extract_image(payload)


# --- Display (native Cocoa, reliable on multi-monitor macOS) ------------------
# OpenCV's HighGUI ignores window coordinates across displays on macOS, so we
# drive an NSWindow directly. NSScreen.frame() and NSWindow.setFrame both use
# the same bottom-left global coordinate space -- no coordinate flipping needed.
_ns = {"app": None, "window": None, "view": None}


def _pick_screen():
    """Return the NSScreen to project on (external preferred), or None."""
    import AppKit
    screens = list(AppKit.NSScreen.screens())
    if not screens:
        return None
    if PROJECTOR_X is not None and PROJECTOR_Y is not None:
        return None  # explicit override handled by caller
    if len(screens) < 2:
        print("  WARNING: only one display detected; using it")
        return screens[0]
    # External = the screen whose frame origin is farthest from (0,0).
    ext = max(screens, key=lambda s: abs(s.frame().origin.x) + abs(s.frame().origin.y))
    return ext


def show_fullscreen(png_bytes: bytes):
    """Show the image fullscreen on the external monitor via a borderless NSWindow."""
    import AppKit
    from Foundation import NSData

    img_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    ns_image = AppKit.NSImage.alloc().initWithData_(img_data)
    if ns_image is None:
        raise RuntimeError("failed to decode generated image for NSImage")

    if _ns["window"] is None:
        screen = _pick_screen()
        if screen is None:
            raise RuntimeError("no display found to project on")
        frame = screen.frame()
        print(f"  projecting on screen frame: origin=({int(frame.origin.x)},"
              f"{int(frame.origin.y)}) size=({int(frame.size.width)},{int(frame.size.height)})")

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)

        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_screen_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
            screen,
        )
        window.setLevel_(AppKit.NSScreenSaverWindowLevel)   # above everything, incl. menu bar
        window.setFrame_display_(frame, True)               # fill the whole external screen

        view = AppKit.NSImageView.alloc().initWithFrame_(window.contentView().bounds())
        # Preserve aspect ratio (letterbox) instead of stretching -- no squished
        # features even when the image aspect != projector aspect.
        view.setImageScaling_(AppKit.NSImageScaleProportionallyUpOrDown)
        view.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        window.setContentView_(view)
        window.makeKeyAndOrderFront_(None)
        window.orderFrontRegardless()
        app.activateIgnoringOtherApps_(True)

        _ns["app"], _ns["window"], _ns["view"] = app, window, view

    _ns["view"].setImage_(ns_image)
    pump_cocoa_events(0.1)


def pump_cocoa_events(seconds: float):
    """Process pending Cocoa events so the window actually draws / stays alive."""
    if _ns["app"] is None:
        return
    import AppKit
    from Foundation import NSDate
    deadline = time.time() + seconds
    while time.time() < deadline:
        ev = _ns["app"].nextEventMatchingMask_untilDate_inMode_dequeue_(
            AppKit.NSEventMaskAny,
            NSDate.dateWithTimeIntervalSinceNow_(0.01),
            AppKit.NSDefaultRunLoopMode,
            True,
        )
        if ev is not None:
            _ns["app"].sendEvent_(ev)


def interactive_loop(regenerate):
    """Keep the projector alive; press 'f' to re-run `regenerate`, 'q'/Ctrl-C to quit.

    Reads single keypresses from the terminal (cbreak mode, no Enter needed).
    Falls back to a plain pump loop if stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        print("(stdin not a TTY -- 'f' disabled; Ctrl-C to exit)")
        try:
            while True:
                pump_cocoa_events(0.2)
        except KeyboardInterrupt:
            pass
        return

    print("Press 'f' to re-capture & re-generate, 'q' or Ctrl-C to quit.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            pump_cocoa_events(0.1)
            if select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1)
                if c in ("q", "\x03"):       # q or Ctrl-C
                    break
                if c == "f":
                    print("\n[f] re-capturing & re-generating...")
                    try:
                        regenerate()
                    except Exception as e:
                        print("regenerate failed:", e)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --- Main ---------------------------------------------------------------------
def one_shot():
    if not ROBOT_TOKEN:
        sys.exit("ROBOT_TOKEN not set (edit the constant at the top)")
    if not AZURE_ENDPOINT:
        sys.exit("AZURE_FLUX_ENDPOINT not set in environment")
    if not AZURE_KEY:
        sys.exit("AZURE_FLUX_KEY not set in environment")

    print("capturing wrist-cam frame...")
    jpg = capture_wrist_frame()
    with open(SOURCE_SNAPSHOT, "wb") as f:
        f.write(jpg)
    print(f"  saved {SOURCE_SNAPSHOT} ({len(jpg)} bytes)")

    print(f"submitting to Azure Flux ({AZURE_MODEL})...")
    t0 = time.time()
    png = run_flux(jpg, PROMPT)
    print(f"  got generated PNG ({len(png)} bytes) in {time.time() - t0:.1f}s")

    with open(OUTPUT_PATH, "wb") as f:
        f.write(png)
    print(f"  saved {OUTPUT_PATH}")

    show_fullscreen(png)
    print("  displayed on external monitor")


def main():
    if LOOP_SECONDS <= 0:
        one_shot()
        interactive_loop(one_shot)
        return

    print(f"looping every {LOOP_SECONDS}s; Ctrl-C to stop.")
    try:
        while True:
            try:
                one_shot()
            except Exception as e:
                print("iteration failed:", e)
            # Keep the window responsive during the wait between generations.
            t_end = time.time() + LOOP_SECONDS
            while time.time() < t_end:
                pump_cocoa_events(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
