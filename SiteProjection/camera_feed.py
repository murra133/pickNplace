"""Continuously grab the wrist-cam feed and overwrite snapshot.jpg every 100 ms.

Run alongside the generator scripts so they always read a fresh frame.
Reuses capture_wrist_frame + the snapshot path from project_flux.

  python3.11 python/camera_feed.py
"""

import os
import time

import project_flux as pf

INTERVAL = 0.1   # seconds between captures (100 ms)
OUT_PATH = pf.SOURCE_SNAPSHOT

# NOTE: darkening now lives in project_flux.capture_wrist_frame() so EVERY
# capture path (camera_feed, depth_flux, project_flux) gets the same correction.
# Tune the exposure there: project_flux.DARKEN_GAIN / DARKEN_GAMMA.


def _atomic_write(path: str, data: bytes) -> None:
    """Write to a temp file then rename, so readers never see a half-written JPEG."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)   # atomic on the same filesystem; overwrites existing


def main():
    if not pf.ROBOT_TOKEN:
        raise SystemExit("ROBOT_TOKEN not set in project_flux.py")

    print(f"saving wrist-cam frames to {OUT_PATH} every {int(INTERVAL * 1000)} ms "
          f"(darken gain={pf.DARKEN_GAIN}, gamma={pf.DARKEN_GAMMA}); Ctrl-C to stop")
    n = 0
    try:
        while True:
            t0 = time.time()
            try:
                jpg = pf.capture_wrist_frame()   # already darkened in project_flux
                _atomic_write(OUT_PATH, jpg)
                n += 1
            except Exception as e:
                print("capture failed:", e)
            # Sleep the remainder of the interval (account for capture time).
            dt = time.time() - t0
            if dt < INTERVAL:
                time.sleep(INTERVAL - dt)
    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")


if __name__ == "__main__":
    main()
