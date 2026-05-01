import argparse
import json
import math
import os
import pickle
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_INPUT_DIR = r"C:\Users\张梓祺\Desktop\PCCD\PCCD"
DEFAULT_OUTPUT_DIR = r"C:\Users\张梓祺\Desktop\PCCD\PCCD_AVA_Format"

COMMENT_FIELDS = [
    "description",
    "general_impression",
    "composition",
    "color_lighting",
    "color",
    "lighting",
    "subject_of_photo",
    "subject",
    "depth_of_field",
    "focus",
    "use_of_camera",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PCCD into the AVA-style layout used by this project."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="PCCD root directory.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the AVA-style PCCD dataset.",
    )
    parser.add_argument(
        "--json-file",
        default="guru.json",
        help="JSON annotation file under input-dir. Defaults to guru.json.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to convert mapped overall scores into 10-bin soft labels.",
    )
    return parser.parse_args()


def load_records(json_path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_overall(value):
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def map_to_1_10(score, min_score, max_score):
    if max_score == min_score:
        return 5.5
    return 1.0 + (score - min_score) * 9.0 / (max_score - min_score)


def gaussian_soft_label(mapped_score, sigma):
    bins = list(range(1, 11))
    weights = [math.exp(-((bin_score - mapped_score) ** 2) / (2.0 * sigma * sigma)) for bin_score in bins]
    total = sum(weights)
    if total <= 0:
        return [0.1] * 10
    return [weight / total for weight in weights]


def build_caption(record):
    parts = []
    seen = set()
    for field in COMMENT_FIELDS:
        text = record.get(field, "")
        if text is None:
            continue
        text = str(text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " ".join(parts)


def save_as_jpg(source_path, target_path):
    try:
        with Image.open(source_path) as image:
            image.convert("RGB").save(target_path, format="JPEG", quality=95)
    except Exception:
        shutil.copy2(source_path, target_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_dir = input_dir / "images" / "full"
    json_path = input_dir / args.json_file

    records = load_records(json_path)
    total_records = len(records)

    parsed_scores = []
    overall_parse_failed = 0
    for record in records:
        score = parse_overall(record.get("overall"))
        if score is None:
            overall_parse_failed += 1
        else:
            parsed_scores.append(score)

    if not parsed_scores:
        raise ValueError("No valid overall scores found in the PCCD JSON file.")

    min_score = min(parsed_scores)
    max_score = max(parsed_scores)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    caption_dict = {}
    missing_images = 0
    empty_captions = 0
    converted = 0

    for record in records:
        title = str(record.get("title", "")).strip()
        overall = parse_overall(record.get("overall"))
        if overall is None:
            continue

        source_path = image_dir / title
        if not title or not source_path.exists():
            missing_images += 1
            continue

        image_id = converted + 1
        target_path = output_dir / "images" / f"{image_id}.jpg"
        save_as_jpg(source_path, target_path)

        caption = build_caption(record)
        if not caption:
            empty_captions += 1
            caption = "no comment"
        caption_dict[image_id] = [caption]

        mapped_score = map_to_1_10(overall, min_score, max_score)
        soft_label = gaussian_soft_label(mapped_score, args.sigma)
        row = {"image_id": image_id}
        for idx, value in enumerate(soft_label, start=2):
            row[f"score{idx}"] = value
        rows.append(row)
        converted += 1

    random.seed(args.seed)
    random.shuffle(rows)
    test_count = int(round(len(rows) * args.test_ratio))
    test_rows = rows[:test_count]
    train_rows = rows[test_count:]

    pd.DataFrame(train_rows).to_csv(output_dir / "train_pccd.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_pccd.csv", index=False)
    with (output_dir / "AVA_Comments_Full.pkl").open("wb") as f:
        pickle.dump(caption_dict, f)

    print(f"Total records: {total_records}")
    print(f"Successfully converted: {converted}")
    print(f"Missing images: {missing_images}")
    print(f"Overall parse failures: {overall_parse_failed}")
    print(f"Empty captions: {empty_captions}")
    print(f"Train count: {len(train_rows)}")
    print(f"Test count: {len(test_rows)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
