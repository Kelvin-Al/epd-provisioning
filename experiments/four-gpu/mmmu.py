"""Prepare a repeatable single-image MMMU serving cohort from cluster_data."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from PIL import Image


def prepare(source, output, calibration, prompts, seed, output_len):
    manifest = json.loads((source / "workload.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != "MMMU/MMMU":
        raise ValueError("Use a manifest prepared from MMMU/MMMU")
    config = json.loads(calibration.read_text(encoding="utf-8"))
    minimum, maximum = config["encoder"]["visual_token_range"]
    rows = [row for row in manifest["requests"] if len(row["images"]) == 1][:prompts]
    if len(rows) != prompts:
        raise ValueError("Too few single-image MMMU requests")
    images = output / "images"
    images.mkdir(parents=True)
    data = []
    resized = 0
    image_hash = sha256()
    for index, row in enumerate(rows):
        with Image.open(source / row["images"][0]) as item:
            image = item.convert("RGB")
        # Qwen2.5-VL patches merge on a 28-pixel grid. Preserve the aspect
        # ratio to within grid rounding while staying inside the fitted range.
        w, h = image.size
        natural = max(1, round(w / 28)) * max(1, round(h / 28))
        if not minimum <= natural <= maximum:
            target = min(max(natural, minimum), maximum)
            narrow = max(1, round(math.sqrt(target * min(w, h) / max(w, h))))
            wide = round(target / narrow)
            wide = min(max(wide, math.ceil(minimum / narrow)), maximum // narrow)
            dimensions = (wide, narrow) if w >= h else (narrow, wide)
            image = image.resize(tuple(side * 28 for side in dimensions), Image.Resampling.LANCZOS)
            resized += 1
        path = images / f"{index:04d}.jpg"
        image.save(path, quality=93)
        image_hash.update(path.read_bytes())
        data.append(json.dumps({"prompt": row["text"], "image_files": [path.as_posix()]}, ensure_ascii=False))
    corpus = output / "requests.jsonl"
    corpus.write_bytes(("\n".join(data) + "\n").encode("utf-8"))
    replay = {
        "dataset_path": corpus.as_posix(),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "split": manifest["split"],
        "source_manifest_sha256": sha256((source / "workload.json").read_bytes()).hexdigest(),
        "cohort_sha256": sha256(corpus.read_bytes()).hexdigest(),
        "image_bytes_sha256": image_hash.hexdigest(),
        "resized_images": resized,
        "visual_token_range": [minimum, maximum],
        "prompts": prompts,
        "output_len": output_len,
        "seed": seed,
    }
    (output / "replay.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=720)
    parser.add_argument("--seed", type=int, default=9197)
    parser.add_argument("--output-len", type=int, default=64)
    args = parser.parse_args()
    if args.output.is_absolute() or min(args.prompts, args.output_len) < 1:
        parser.error("Use a relative output directory and positive request counts")
    prepare(args.source, args.output, args.calibration, args.prompts, args.seed, args.output_len)
