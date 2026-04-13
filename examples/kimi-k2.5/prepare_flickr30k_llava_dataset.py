#!/usr/bin/env python3
"""Build a small LLaVA-style Flickr30k dataset for Kimi-K2.5-VL sanity checks.

This script downloads `lmms-lab/flickr30k` from Hugging Face, materializes the
images locally, and emits two dataset views:

1. LLaVA-style JSON files with `image` + `conversations`.
2. Simple JSONL files with `image` + `prompt` + `text`, which can be consumed
   directly by `pretrain_kimi_k25_vl.sh`.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_DATASET_NAME = "lmms-lab/flickr30k"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_PROMPT = "Describe the image in detail."


@dataclass(frozen=True)
class SplitSpec:
    name: str
    image_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert lmms-lab/flickr30k into a LLaVA-style dataset and a "
            "simple image/prompt/text JSONL dataset for Kimi-K2.5-VL sanity checks."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the converted dataset into.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT, help="Hugging Face split to read from.")
    parser.add_argument("--train-image-count", type=int, default=1024, help="Number of source images to place in the train split.")
    parser.add_argument("--valid-image-count", type=int, default=128, help="Number of source images to place in the valid split.")
    parser.add_argument("--test-image-count", type=int, default=128, help="Number of source images to place in the test split.")
    parser.add_argument(
        "--captions-per-image",
        type=str,
        choices=("all", "first", "random"),
        default="all",
        help=(
            "How many captions to emit for each source image. "
            "`all` expands each image into multiple samples."
        ),
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text used in generated samples.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split shuffling and caption sampling.")
    parser.add_argument("--cache-dir", type=Path, default="/home/dist/cchen/data/cache", help="Optional Hugging Face datasets cache directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing new files.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency. Install with `pip install datasets pillow`."
        ) from exc


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def load_source_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.dataset_split,
    }
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    return load_dataset(**load_kwargs)


def validate_counts(total_images: int, split_specs: Sequence[SplitSpec]) -> None:
    requested = sum(spec.image_count for spec in split_specs)
    if requested <= 0:
        raise SystemExit("At least one output split must contain images.")
    if requested > total_images:
        raise SystemExit(
            f"Requested {requested} images but source split only contains {total_images} images."
        )


def shuffled_indices(total_images: int, seed: int) -> List[int]:
    indices = list(range(total_images))
    random.Random(seed).shuffle(indices)
    return indices


def select_captions(captions: Sequence[str], mode: str, rng: random.Random) -> Iterable[tuple[int, str]]:
    normalized = [str(caption).strip() for caption in captions if str(caption).strip()]
    if not normalized:
        raise ValueError("Encountered a sample without any non-empty captions.")

    if mode == "all":
        return list(enumerate(normalized))
    if mode == "first":
        return [(0, normalized[0])]
    if mode == "random":
        index = rng.randrange(len(normalized))
        return [(index, normalized[index])]
    raise ValueError(f"Unsupported captions-per-image mode: {mode}")


def image_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        return suffix
    return ".jpg"


def save_image(image_obj: Any, output_path: Path) -> None:
    from PIL import Image

    if isinstance(image_obj, dict):
        image_path = image_obj.get("path")
        image_bytes = image_obj.get("bytes")
        if image_path:
            shutil.copy2(image_path, output_path)
            return
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(output_path)
            return

    if hasattr(image_obj, "convert"):
        image = image_obj.convert("RGB")
        image.save(output_path)
        return

    raise TypeError(f"Unsupported image object type: {type(image_obj)!r}")


def relative_image_path(split_name: str, filename: str) -> Path:
    safe_name = Path(filename).name
    return Path("images") / split_name / safe_name


def build_llava_record(sample_id: str, rel_image_path: Path, prompt: str, answer: str) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "image": rel_image_path.as_posix(),
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_simple_record(
    sample_id: str,
    rel_image_path: Path,
    prompt: str,
    answer: str,
    source_row_index: int,
    source_caption_index: int,
    filename: str,
    img_id: Any,
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "image": rel_image_path.as_posix(),
        "prompt": prompt,
        "text": answer,
        "source_row_index": source_row_index,
        "source_caption_index": source_caption_index,
        "source_filename": filename,
        "source_img_id": img_id,
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_split(
    dataset,
    split_spec: SplitSpec,
    indices: Sequence[int],
    output_dir: Path,
    prompt: str,
    captions_per_image: str,
    seed: int,
) -> Dict[str, int]:
    split_rng = random.Random(seed)
    split_image_dir = output_dir / "images" / split_spec.name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    llava_records: List[Dict[str, Any]] = []
    simple_records: List[Dict[str, Any]] = []

    for output_row_index, dataset_index in enumerate(indices):
        row = dataset[dataset_index]
        captions = row["caption"]
        filename = str(row.get("filename") or f"{row.get('img_id', dataset_index)}{image_suffix('')}")
        rel_image_path = relative_image_path(split_spec.name, filename)
        abs_image_path = output_dir / rel_image_path
        if not abs_image_path.exists():
            save_image(row["image"], abs_image_path)

        for caption_index, caption in select_captions(captions, captions_per_image, split_rng):
            sample_id = f"flickr30k_{split_spec.name}_{dataset_index:05d}_{caption_index:02d}"
            llava_records.append(build_llava_record(sample_id, rel_image_path, prompt, caption))
            simple_records.append(
                build_simple_record(
                    sample_id=sample_id,
                    rel_image_path=rel_image_path,
                    prompt=prompt,
                    answer=caption,
                    source_row_index=dataset_index,
                    source_caption_index=caption_index,
                    filename=filename,
                    img_id=row.get("img_id"),
                )
            )

        if (output_row_index + 1) % 100 == 0:
            print(
                f"[{split_spec.name}] materialized {output_row_index + 1}/{len(indices)} images, "
                f"{len(simple_records)} samples"
            )

    write_json(output_dir / f"{split_spec.name}.llava.json", llava_records)
    write_jsonl(output_dir / f"{split_spec.name}.simple.jsonl", simple_records)

    return {
        "images": len(indices),
        "samples": len(simple_records),
    }


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    split_stats: Dict[str, Dict[str, int]],
    source_image_count: int,
) -> None:
    metadata = {
        "source_dataset": args.dataset_name,
        "source_config": args.dataset_config,
        "source_split": args.dataset_split,
        "source_image_count": source_image_count,
        "prompt": args.prompt,
        "captions_per_image": args.captions_per_image,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "llava_train": "train.llava.json",
            "llava_valid": "valid.llava.json",
            "llava_test": "test.llava.json",
            "simple_train": "train.simple.jsonl",
            "simple_valid": "valid.simple.jsonl",
            "simple_test": "test.simple.jsonl",
        },
        "split_stats": split_stats,
    }
    write_json(output_dir / "dataset_info.json", metadata)


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    prepare_output_dir(args.output_dir, args.overwrite)

    dataset = load_source_dataset(args)
    split_specs = (
        SplitSpec("train", args.train_image_count),
        SplitSpec("valid", args.valid_image_count),
        SplitSpec("test", args.test_image_count),
    )
    validate_counts(len(dataset), split_specs)

    shuffled = shuffled_indices(len(dataset), args.seed)
    offset = 0
    split_stats: Dict[str, Dict[str, int]] = {}
    for split_spec in split_specs:
        split_indices = shuffled[offset : offset + split_spec.image_count]
        offset += split_spec.image_count
        split_stats[split_spec.name] = build_split(
            dataset=dataset,
            split_spec=split_spec,
            indices=split_indices,
            output_dir=args.output_dir,
            prompt=args.prompt.strip(),
            captions_per_image=args.captions_per_image,
            seed=args.seed + offset,
        )

    write_metadata(args.output_dir, args, split_stats, len(dataset))

    print("Finished dataset conversion.")
    for split_name, stats in split_stats.items():
        print(
            f"  {split_name}: {stats['images']} images -> {stats['samples']} samples"
        )
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
