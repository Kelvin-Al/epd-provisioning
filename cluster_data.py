"""Prepare a local image workload from public dataset releases."""

import ast
import itertools
import json
import re

DATASETS = {
    "mmmu": ("MMMU/MMMU", "validation"),
    "vqav2": ("lmms-lab-encoder/VQAv2", "validation"),
    "textvqa": ("lmms-lab-encoder/textvqa", "validation"),
}
IMAGE_MARKER = re.compile(r"<image\s+(\d+)>")


def normalize(dataset, row):
    question = row["question"].strip()
    if dataset == "mmmu":
        found = list(dict.fromkeys(int(i) for i in IMAGE_MARKER.findall(question)))
        available = {i: row[f"image_{i}"] for i in range(1, 8) if row.get(f"image_{i}")}
        if any(i not in available for i in found):
            return None
        images = [available[i] for i in found or sorted(available)]
        if found:
            positions = {image: index + 1 for index, image in enumerate(found)}
            question = IMAGE_MARKER.sub(
                lambda match: f"[Image {positions[int(match[1])]}]", question
            )
        choices = row.get("options") or []
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        if choices:
            question += "\n" + "\n".join(
                f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
            )
            question += "\nAnswer with the option letter."
    else:
        images = [row["image"]] if row.get("image") is not None else []
        question += "\nAnswer briefly."
    return (question, images) if question and images else None


def prepare(args):
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi

    repo, default_split = DATASETS[args.dataset]
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    split = args.split or default_split
    subsets = args.subset or (
        sorted(get_dataset_config_names(repo, revision=revision))
        if args.dataset == "mmmu" else [None]
    )
    args.output.mkdir()
    (args.output / "images").mkdir()
    readers = [
        iter(load_dataset(repo, name, split=split, revision=revision, streaming=True).shuffle(
            seed=args.seed, buffer_size=args.shuffle_buffer
        ))
        for name in subsets
    ]
    rows = []
    for group in itertools.zip_longest(*readers):
        for name, row in zip(subsets, group):
            if row is None:
                continue
            normalized = normalize(args.dataset, row)
            if normalized is None:
                continue
            question, images = normalized
            if len(images) > args.max_images or any(
                image.width * image.height > args.max_pixels for image in images
            ):
                continue
            paths = []
            for index, image in enumerate(images):
                path = f"images/{len(rows):06d}-{index}.png"
                image.convert("RGB").save(args.output / path)
                paths.append(path)
            rows.append({"text": question, "images": paths, "subset": name})
            if len(rows) == args.samples:
                break
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        raise ValueError(f"only {len(rows)} eligible rows available for {args.samples} requested")
    manifest = {
        "dataset": repo, "revision": revision, "split": split, "subsets": subsets,
        "seed": args.seed, "shuffle_buffer": args.shuffle_buffer,
        "max_images": args.max_images, "max_pixels": args.max_pixels, "requests": rows,
    }
    (args.output / "workload.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_workload(root):
    manifest = json.loads((root / "workload.json").read_text(encoding="utf-8"))
    if not manifest["requests"]:
        raise ValueError("workload contains no requests")
    root = root.resolve()
    workload = []
    for row in manifest["requests"]:
        images = [(root / name).resolve() for name in row["images"]]
        if not images or any(
            not path.is_relative_to(root) or not path.is_file() for path in images
        ):
            raise ValueError("workload images must exist inside the workload directory")
        workload.append((row["text"], images))
    return workload
