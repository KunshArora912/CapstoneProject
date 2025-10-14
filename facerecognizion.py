"""
faces_app.py — one-file face encoding + live camera recognition

Usage (after `pip install -r requirements.txt`):

# 1) Prepare your dataset (one folder per person):
#    data/known/Ada_Lovelace/img1.jpg, img2.jpg ...
#    data/known/Grace_Hopper/img1.jpg, img2.jpg ...

# 2) Encode faces (creates data/encodings.pickle):
python faces_app.py encode

# 3) Run live camera recognition:
python faces_app.py live

# Optional flags:
#   --model {hog,cnn}          # detection model (cnn needs CUDA build of dlib for speed)
#   --downscale 0.25           # shrink frames for speed (0.25-0.5 common)
#   --tolerance 0.45           # lower = stricter, typical 0.4-0.6
#   --upsample 0               # 0-2 (detect smaller faces with higher values)
#   --jitters 1                # 1-10 (more robust encodings, slower)
#   --every 2                  # process every Nth frame
#   --cam 0                    # camera index (0/1/2...)
#   --save-unknowns            # save crops of "Unknown" faces to data/unknowns/

# 4) Test on a still image (draws boxes/labels and writes to output/):
python faces_app.py image -f path/to/photo.jpg

Data dirs auto-create as needed: data/known/, data/unknowns/, output/
"""

import os
import time
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np
import face_recognition
import argparse


# ---------- Paths ----------
DATA_DIR = Path("data")
KNOWN_DIR = DATA_DIR / "known"
UNKNOWN_DIR = DATA_DIR / "unknowns"
ENCODINGS_PATH = DATA_DIR / "encodings.pickle"
OUTPUT_DIR = Path("output")

# ---------- Defaults / Tunables ----------
DEFAULTS = dict(
    model="hog",        # 'hog' (CPU) or 'cnn' (GPU build recommended)
    downscale=0.25,     # 0.25..1.0
    tolerance=0.45,     # 0.4..0.6 typical
    upsample=0,         # 0..2
    jitters=1,          # 1..10
    every=2,            # process every Nth frame
    cam=0,              # camera index
    vote_top_k=3,       # majority vote over k closest matches
)

BOX_COLOR = (0, 255, 0)      # BGR for rectangles
TEXT_COLOR = (255, 255, 255) # white text
TEXT_BG = (0, 0, 0)          # black label background


# ---------- Utils ----------
def ensure_dirs():
    KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def draw_label(frame, label, left, bottom):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (left, bottom), (left + tw + 10, bottom + th + 10), TEXT_BG, -1)
    cv2.putText(frame, label, (left + 5, bottom + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)


def annotate(frame, name, top, right, bottom, left):
    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, 2)
    draw_label(frame, name, left, bottom)


def save_unknown_crop(frame, box):
    top, right, bottom, left = box
    h, w = frame.shape[:2]
    # safe clamp
    top = max(0, top); left = max(0, left)
    bottom = min(h, bottom); right = min(w, right)
    crop = frame[top:bottom, left:right]
    ts = int(time.time() * 1000)
    out = UNKNOWN_DIR / f"unknown_{ts}.jpg"
    cv2.imwrite(str(out), crop)


def encode_single_image(img_path: Path, model: str, upsample: int, jitters: int) -> List[np.ndarray]:
    image = face_recognition.load_image_file(str(img_path))
    boxes = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model=model)
    if not boxes:
        print(f"[WARN] No face in {img_path}")
        return []
    # if multiple faces: pick largest
    if len(boxes) > 1:
        def area(b): t, r, btm, l = b; return (r - l) * (btm - t)
        boxes = [max(boxes, key=area)]
    encs = face_recognition.face_encodings(image, known_face_locations=boxes, num_jitters=jitters)
    return encs


def encode_dataset(model: str, upsample: int, jitters: int):
    assert KNOWN_DIR.exists(), f"Missing {KNOWN_DIR}. Put person folders inside."
    names, encs = [], []
    people_dirs = [p for p in KNOWN_DIR.iterdir() if p.is_dir()]
    if not people_dirs:
        raise SystemExit(f"No person folders in {KNOWN_DIR}. Create e.g. {KNOWN_DIR}/Alice/, {KNOWN_DIR}/Bob/")

    for person_dir in sorted(people_dirs):
        person = person_dir.name
        imgs = list_images(person_dir)
        if not imgs:
            print(f"[WARN] No images for {person}")
            continue
        print(f"[INFO] Encoding {person} ({len(imgs)} images)...")
        cnt = 0
        for img in imgs:
            encs_list = encode_single_image(img, model, upsample, jitters)
            if encs_list:
                encs.append(encs_list[0])
                names.append(person)
                cnt += 1
        print(f"[INFO] -> kept {cnt} encodings for {person}")
    if not encs:
        raise SystemExit("No encodings produced. Check your images and try again.")
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": encs, "names": names}, f)
    print(f"[DONE] Saved {len(encs)} encodings for {len(set(names))} people -> {ENCODINGS_PATH}")


def load_db(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Missing encodings: {path}. Run `python faces_app.py encode` first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    known_encs = np.array(data["encodings"])
    known_names = np.array(data["names"])
    return known_encs, known_names


def majority_vote(candidates, k=3) -> str:
    if not candidates:
        return "Unknown"
    top = [n for n, _d in candidates[:k]]
    return Counter(top).most_common(1)[0][0]


def match_face(known_encs, known_names, enc, tolerance: float, vote_top_k: int) -> str:
    dists = face_recognition.face_distance(known_encs, enc)
    idxs = np.where(dists <= tolerance)[0]
    if idxs.size == 0:
        return "Unknown"
    sorted_idxs = idxs[np.argsort(dists[idxs])]
    candidates = [(known_names[i], float(dists[i])) for i in sorted_idxs]
    return majority_vote(candidates, k=vote_top_k)


def upscale_boxes(boxes_small, scale: float):
    boxes = []
    for (top, right, bottom, left) in boxes_small:
        boxes.append((
            int(top / scale),
            int(right / scale),
            int(bottom / scale),
            int(left / scale),
        ))
    return boxes


def recognize_in_frame(
    frame_bgr,
    known_encs,
    known_names,
    model: str,
    downscale: float,
    upsample: int,
    jitters: int,
    tolerance: float,
    vote_top_k: int
):
    small = cv2.resize(frame_bgr, (0, 0), fx=downscale, fy=downscale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes_small = face_recognition.face_locations(
        rgb_small, number_of_times_to_upsample=upsample, model=model
    )
    encs_small = face_recognition.face_encodings(
        rgb_small, known_face_locations=boxes_small, num_jitters=jitters
    )
    boxes = upscale_boxes(boxes_small, downscale)
    result = []
    for enc, box in zip(encs_small, boxes):
        name = match_face(known_encs, known_names, enc, tolerance, vote_top_k=vote_top_k)
        result.append((name, box))
    return result


# ---------- Commands ----------
def cmd_encode(args):
    ensure_dirs()
    encode_dataset(model=args.model, upsample=args.upsample, jitters=args.jitters)


def cmd_live(args):
    ensure_dirs()
    known_encs, known_names = load_db(ENCODINGS_PATH)
    print(f"[INFO] DB: {len(known_encs)} encodings across {len(set(known_names))} people")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.cam}. Try --cam 1 or check permissions.")

    frame_idx = 0
    fps = 0.0
    last_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                break

            display = frame.copy()
            if frame_idx % args.every == 0:
                matches = recognize_in_frame(
                    frame_bgr=frame,
                    known_encs=known_encs,
                    known_names=known_names,
                    model=args.model,
                    downscale=args.downscale,
                    upsample=args.upsample,
                    jitters=args.jitters,
                    tolerance=args.tolerance,
                    vote_top_k=DEFAULTS["vote_top_k"],
                )
                last_matches = matches
            else:
                # hold previous results on skipped frames
                matches = last_matches if 'last_matches' in locals() else []

            for name, (top, right, bottom, left) in matches:
                annotate(display, name, top, right, bottom, left)
                if args.save_unknowns and name == "Unknown":
                    save_unknown_crop(frame, (top, right, bottom, left))

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_t = now
            draw_label(display, f"FPS: {fps:.1f}", 10, 25)

            cv2.imshow("Face Recognition (q to quit)", display)
            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cmd_image(args):
    ensure_dirs()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"Image not found: {path}")
    known_encs, known_names = load_db(ENCODINGS_PATH)

    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {path}")

    matches = recognize_in_frame(
        frame_bgr=frame,
        known_encs=known_encs,
        known_names=known_names,
        model=args.model,
        downscale=args.downscale,
        upsample=args.upsample,
        jitters=args.jitters,
        tolerance=args.tolerance,
        vote_top_k=DEFAULTS["vote_top_k"],
    )
    for name, (top, right, bottom, left) in matches:
        annotate(frame, name, top, right, bottom, left)

    out = OUTPUT_DIR / f"{path.stem}_labeled.jpg"
    cv2.imwrite(str(out), frame)
    print(f"[DONE] Wrote {out}")


# ---------- CLI ----------
def build_parser():
    p = argparse.ArgumentParser(description="One-file Face Recognition (encode + live + image test)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--model", choices=["hog", "cnn"], default=DEFAULTS["model"])
        sp.add_argument("--downscale", type=float, default=DEFAULTS["downscale"])
        sp.add_argument("--tolerance", type=float, default=DEFAULTS["tolerance"])
        sp.add_argument("--upsample", type=int, default=DEFAULTS["upsample"])
        sp.add_argument("--jitters", type=int, default=DEFAULTS["jitters"])

    sp_encode = sub.add_parser("encode", help="Encode faces from data/known/* folders")
    add_common(sp_encode)
    sp_encode.set_defaults(func=cmd_encode)

    sp_live = sub.add_parser("live", help="Run live camera recognition")
    add_common(sp_live)
    sp_live.add_argument("--every", type=int, default=DEFAULTS["every"])
    sp_live.add_argument("--cam", type=int, default=DEFAULTS["cam"])
    sp_live.add_argument("--save-unknowns", action="store_true")
    sp_live.set_defaults(func=cmd_live)

    sp_image = sub.add_parser("image", help="Run recognition on a single image and save to output/")
    add_common(sp_image)
    sp_image.add_argument("-f", "--file", required=True, help="Path to an image file")
    sp_image.set_defaults(func=cmd_image)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
