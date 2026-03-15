#!/usr/bin/env python3
"""Run one image through the exact backend inference path and save annotated output."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

import server


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one frame through detector and save annotated result."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input_stem>_annotated.jpg",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save detection JSON",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=95,
        help="JPEG quality used when sending the frame to the model",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Override server MIN_TRACKING_SCORE for this run. "
            "Lower for debugging if detections are missing."
        ),
    )
    return parser.parse_args()


def clamp(n: float, lo: float, hi: float) -> float:
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def draw_detections(
    image: Image.Image,
    detections: List[Dict],
) -> None:
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for det in detections:
        x, y, bw, bh = det["bbox"]
        x2 = x + bw
        y2 = y + bh

        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)
        if x2 <= x or y2 <= y:
            continue

        is_healthy = det.get("in_schijf_van_vijf", True)
        color = "green" if is_healthy else "red"
        draw.rectangle([x, y, x2, y2], outline=color, width=3)

        label = f"{det['class']} | {det['confidence']:.2f}" + (
            f" | id {det['tracker_id']}" if "tracker_id" in det else ""
        )
        draw.text((x + 2, y + 2), label, fill=color)


def run_raw_inference(image: Image.Image, min_score: float) -> List[Dict]:
    sv_dets = server.run_model_inference(image, min_score)
    if len(sv_dets) == 0:
        return []

    detections = []
    for i in range(len(sv_dets)):
        detections.append(
            server.build_detection(
                sv_dets.xyxy[i],
                int(sv_dets.class_id[i]),
                float(sv_dets.confidence[i]),
            )
        )
    return detections


def main():
    args = parse_args()
    input_path = Path(args.image).expanduser().resolve()

    image = Image.open(input_path).convert("RGB")
    with BytesIO() as out:
        image.save(out, format="JPEG", quality=args.max_size, optimize=True)
        jpeg_bytes = out.getvalue()

    image_for_inference = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    min_score = server.MIN_TRACKING_SCORE if args.min_score is None else args.min_score
    detections = run_raw_inference(image_for_inference, min_score=min_score)
    draw_detections(image, detections)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_annotated.jpg")
    )
    image.save(output_path, quality=95)

    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        json_path.write_text(json.dumps({"detections": detections}, indent=2))

    print(f"Saved annotated image: {output_path}")
    print(f"Detections: {len(detections)}")


if __name__ == "__main__":
    main()
