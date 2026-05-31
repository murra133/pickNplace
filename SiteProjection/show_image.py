"""Open an image fullscreen on the second (external) monitor. No AI.

  python3.11 python/show_image.py [path]

Defaults to generated1.png. Press Ctrl-C to exit.
Reuses the native-Cocoa fullscreen display from project_flux.
"""

import sys

import project_flux as pf


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "generated1.png"
    with open(path, "rb") as f:
        data = f.read()

    pf.GRID_QUAD = None          # show the image as-is (no grid masking)
    pf.show_fullscreen(data)
    print(f"showing {path} on external monitor. Ctrl-C to shut off.")
    try:
        while True:
            pf.pump_cocoa_events(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        pf.close_display()       # close the projector window on shutdown
        print("\nimage share off.")


if __name__ == "__main__":
    main()
