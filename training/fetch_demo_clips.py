"""Cut short real dashcam demo clips (Wikimedia Commons, free licences) into demo/videos/ for the dashboard.

Called at the end of the training notebook, or on its own:
    docker compose --profile train run --rm trainer python fetch_demo_clips.py

Only the needed part of each remote video is downloaded (ffmpeg seeks with HTTP range requests).
Clips that fail (e.g. Wikimedia rate limit, HTTP 429) are skipped; run the script again later.
"""
import subprocess
import time
from pathlib import Path

import cv2

DEMO = Path("/workspace/demo/videos")
TEST_IMAGES = Path("/datasets/road_hazards/images/test")
UA = ["-user_agent", "iot-smartcar-training/1.0 (student project; demo clips)"]   # Wikimedia asks for a real UA

CLIPS = [  # output, source URL, start (s), duration (s), credit / licence
    ("01_city_krakow.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/9/98/City_Driving_4K-_Krak%C3%B3w_Poland_2024.webm",
     360, 75, "City Driving 4K – Kraków Poland 2024 · Relaxing Roads · CC BY 3.0"),
    ("02_barricades.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/8/8d/Dash_cam_footage_shows_driver_ignoring_barricades%2C_nearly_hitting_people.webm",
     20, 50, "Dash cam footage shows driver ignoring barricades · Devinn Taufner · Public domain"),
    ("03_lane_blocked.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/c/cd/Connect_to_the_opposite_lane_and_bypass_the_traffic_accident_section.webm",
     0, 30, "Bypass of a traffic accident section · Public domain"),
    ("04_pothole_street.mp4", "https://upload.wikimedia.org/wikipedia/commons/d/d0/Cars_Moving_into_pothole.webm",
     0, 21, "Cars Moving into pothole · Amuzujoe · CC BY-SA 4.0"),
    ("04b_pothole_kumasi.mp4", "https://upload.wikimedia.org/wikipedia/commons/5/58/Pothole.ogv",
     0, 14, "Pothole (Kumasi) · Amuzujoe · CC BY-SA 4.0"),
]
SLIDESHOW = ("05_potholes_test_frames.mp4", "held-out test frames of the Potholes dataset "
             "(Bučko et al., Sensors 2022) · CC BY 4.0")


def ffmpeg(args, attempts=4):
    for i in range(attempts):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"  ffmpeg failed ({r.stderr.strip()[:120]}), retry in {30 * (i + 1)}s", flush=True)
        time.sleep(30 * (i + 1))
    return False


def cut(src, dst, start, duration):
    return ffmpeg((["-ss", str(start)] if start else []) + UA +
                  ["-i", src, "-t", str(duration), "-vf", "scale=-2:720,fps=15", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(dst)])


def pothole_slideshow(out, n=60, seconds_each=0.7, fps=15):
    """Real dashcam pothole frames the model never saw (test split), as a short video."""
    frames = sorted(TEST_IMAGES.glob("pothole_*.jpg"))[::2][:n]
    if not frames:
        return False
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for p in frames:
        img = cv2.resize(cv2.imread(str(p)), (1280, 720))
        for _ in range(round(seconds_each * fps)):
            writer.write(img)
    writer.release()
    ok = ffmpeg(["-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], attempts=1)
    tmp.unlink(missing_ok=True)
    return ok


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    done = []
    for name, url, start, dur, credit in CLIPS:
        out = DEMO / name
        if not out.exists():
            print("cutting", name, "…", flush=True)
            if not cut(url, out, start, dur):
                out.unlink(missing_ok=True)
                print("  skipped", name, "(run this script again later)", flush=True)
                continue
        done.append((name, credit))
    name, credit = SLIDESHOW
    if (DEMO / name).exists() or pothole_slideshow(DEMO / name):
        done.append((name, credit))
    lines = ["# Demo clips — sources and licences", ""] + [f"- `{n}` — {c}" for n, c in done]
    (DEMO / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return done


if __name__ == "__main__":
    main()
