#!/usr/bin/env python3
"""Convert a local Flickr30k download into Kimi-K2.5-VL training files.

The expected source directory is the layout produced by:

    hf download nlphuji/flickr30k --repo-type dataset --local-dir data/flickr30k

Required files:
  - flickr_annotations_30k.csv
  - flickr30k-images.zip, or an extracted flickr30k-images/ directory

Outputs:
  - train.simple.jsonl / valid.simple.jsonl / test.simple.jsonl
  - train.llava.json / valid.llava.json / test.llava.json unless disabled
  - images/<split>/<filename>
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO


DEFAULT_SOURCE_DIR = Path("data/flickr30k")
DEFAULT_PROMPT = "Describe the image in detail."
OUTPUT_SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {
    "train": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "test": "test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class Annotation:
    row_index: int
    source_split: str
    output_split: str
    filename: str
    img_id: str
    captions: Sequence[str]


@dataclass(frozen=True)
class ImageSource:
    kind: str
    path: Path
    index: Dict[str, str]


class JsonArrayWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Optional[TextIO] = None
        self.first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.handle = self.path.open("w", encoding="utf-8")
        self.handle.write("[")
        return self

    def write(self, record: Dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JsonArrayWriter is not open")
        if self.first:
            self.handle.write("\n")
            self.first = False
        else:
            self.handle.write(",\n")
        self.handle.write(json.dumps(record, ensure_ascii=False))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is None:
            return
        if not self.first:
            self.handle.write("\n")
        self.handle.write("]\n")
        self.handle.close()


class NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class ImageMaterializer:
    def __init__(self, image_source: ImageSource):
        self.image_source = image_source
        self.zip_handle: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> "ImageMaterializer":
        if self.image_source.kind == "zip":
            self.zip_handle = zipfile.ZipFile(self.image_source.path)
        return self

    def copy_to(self, filename: str, destination: Path) -> None:
        if destination.exists():
            return
        source_ref = self.image_source.index.get(filename)
        if source_ref is None:
            raise FileNotFoundError(f"image not found in source: {filename}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f".{destination.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()

        if self.image_source.kind == "dir":
            shutil.copy2(source_ref, temp_path)
        else:
            if self.zip_handle is None:
                raise RuntimeError("zip image source is not open")
            with self.zip_handle.open(source_ref) as reader, temp_path.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
        temp_path.replace(destination)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.zip_handle is not None:
            self.zip_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the local nlphuji/flickr30k download into full Kimi-K2.5-VL "
            "JSONL files using the dataset's train/val/test split."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Local Flickr30k directory downloaded by `hf download`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write converted JSONL files and materialized images.",
    )
    parser.add_argument(
        "--image-source",
        type=Path,
        default=None,
        help=(
            "Optional extracted image directory or image zip. By default the script "
            "uses <source-dir>/flickr30k-images or <source-dir>/flickr30k-images.zip."
        ),
    )
    parser.add_argument(
        "--captions-per-image",
        choices=("all", "first"),
        default="all",
        help="Use every caption as a sample, or only the first caption for each image.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text used in generated samples.")
    parser.add_argument(
        "--skip-llava-json",
        action="store_true",
        help="Only write the Megatron simple JSONL files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing new files.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def prepare_output_dir(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    source_resolved = source_dir.resolve(strict=False)
    output_resolved = output_dir.resolve(strict=False)
    if overwrite and is_relative_to(source_resolved, output_resolved):
        raise SystemExit(
            f"Refusing to overwrite {output_dir}: it contains the source dataset {source_dir}."
        )

    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def parse_caption_list(raw_value: str, filename: str) -> List[str]:
    value = raw_value.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)

    if isinstance(parsed, str):
        captions = [parsed.strip()]
    elif isinstance(parsed, list):
        captions = [str(caption).strip() for caption in parsed if str(caption).strip()]
    else:
        raise ValueError(f"unsupported caption payload for {filename}: {type(parsed)!r}")

    if not captions:
        raise ValueError(f"no non-empty captions for {filename}")
    return captions


def load_annotations(csv_path: Path) -> List[Annotation]:
    if not csv_path.is_file():
        raise SystemExit(f"Missing annotation CSV: {csv_path}")

    annotations: List[Annotation] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"raw", "split", "filename", "img_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            filename = Path(row["filename"].strip()).name
            source_split = row["split"].strip()
            output_split = SPLIT_ALIASES.get(source_split.lower())
            if output_split is None:
                raise ValueError(f"unsupported split {source_split!r} at row {row_index}")
            annotations.append(
                Annotation(
                    row_index=row_index,
                    source_split=source_split,
                    output_split=output_split,
                    filename=filename,
                    img_id=str(row["img_id"]).strip(),
                    captions=parse_caption_list(row["raw"], filename),
                )
            )
    return annotations


def build_directory_image_source(path: Path) -> ImageSource:
    index: Dict[str, str] = {}
    for image_path in path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(image_path.name, str(image_path))
    return ImageSource(kind="dir", path=path, index=index)


def build_zip_image_source(path: Path) -> ImageSource:
    index: Dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            name = Path(info.filename).name
            if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(name, info.filename)
    return ImageSource(kind="zip", path=path, index=index)


def detect_image_source(source_dir: Path, image_source_arg: Optional[Path]) -> ImageSource:
    candidates = [image_source_arg] if image_source_arg is not None else [
        source_dir / "flickr30k-images",
        source_dir / "images",
        source_dir / "flickr30k-images.zip",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.is_dir():
            return build_directory_image_source(candidate)
        if candidate.is_file() and zipfile.is_zipfile(candidate):
            return build_zip_image_source(candidate)

    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise SystemExit(f"Could not find a Flickr30k image directory or zip. Searched: {searched}")


def validate_images(annotations: Sequence[Annotation], image_source: ImageSource) -> None:
    required = {annotation.filename for annotation in annotations}
    missing = sorted(required - set(image_source.index))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise SystemExit(f"Missing images in {image_source.path}: {preview}{suffix}")


def iter_captions(captions: Sequence[str], mode: str) -> Iterable[tuple[int, str]]:
    if mode == "first":
        return [(0, captions[0])]
    if mode == "all":
        return list(enumerate(captions))
    raise ValueError(f"unsupported captions-per-image mode: {mode}")


def sample_id(annotation: Annotation, caption_index: int) -> str:
    image_id = annotation.img_id or Path(annotation.filename).stem
    return f"flickr30k_{annotation.output_split}_{image_id}_{caption_index:02d}"


def relative_image_path(split_name: str, filename: str) -> Path:
    return Path("images") / split_name / Path(filename).name


def build_simple_record(sample_id_value: str, rel_image_path: Path, prompt: str, answer: str) -> Dict[str, Any]:
    return {
        "id": sample_id_value,
        "image": rel_image_path.as_posix(),
        "prompt": prompt,
        "text": answer,
    }


def build_llava_record(sample_id_value: str, rel_image_path: Path, prompt: str, answer: str) -> Dict[str, Any]:
    return {
        "id": sample_id_value,
        "image": rel_image_path.as_posix(),
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_split(
    split_name: str,
    annotations: Sequence[Annotation],
    materializer: ImageMaterializer,
    output_dir: Path,
    prompt: str,
    captions_per_image: str,
    write_llava: bool,
) -> Dict[str, Any]:
    simple_path = output_dir / f"{split_name}.simple.jsonl"
    llava_path = output_dir / f"{split_name}.llava.json"
    llava_context = JsonArrayWriter(llava_path) if write_llava else None

    image_count = 0
    sample_count = 0
    with simple_path.open("w", encoding="utf-8") as simple_handle:
        with llava_context if llava_context is not None else NullContext():
            for annotation in annotations:
                rel_image_path = relative_image_path(split_name, annotation.filename)
                materializer.copy_to(annotation.filename, output_dir / rel_image_path)
                image_count += 1

                for caption_index, caption in iter_captions(annotation.captions, captions_per_image):
                    current_id = sample_id(annotation, caption_index)
                    simple_handle.write(
                        json.dumps(
                            build_simple_record(current_id, rel_image_path, prompt, caption),
                            ensure_ascii=False,
                        )
                    )
                    simple_handle.write("\n")
                    if llava_context is not None:
                        llava_context.write(build_llava_record(current_id, rel_image_path, prompt, caption))
                    sample_count += 1

                if image_count % 1000 == 0:
                    print(f"[{split_name}] wrote {image_count} images, {sample_count} samples")

    stats: Dict[str, Any] = {
        "images": image_count,
        "samples": sample_count,
        "simple_jsonl": simple_path.name,
    }
    if write_llava:
        stats["llava_json"] = llava_path.name
    return stats


def write_metadata(
    output_dir: Path,
    source_dir: Path,
    image_source: ImageSource,
    prompt: str,
    captions_per_image: str,
    write_llava: bool,
    split_stats: Dict[str, Dict[str, Any]],
) -> None:
    metadata = {
        "source_dataset": "nlphuji/flickr30k",
        "source_dir": str(source_dir),
        "annotation_csv": str(source_dir / "flickr_annotations_30k.csv"),
        "image_source": str(image_source.path),
        "image_source_type": image_source.kind,
        "prompt": prompt,
        "captions_per_image": captions_per_image,
        "write_llava_json": write_llava,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": split_stats,
    }
    write_json(output_dir / "dataset_info.json", metadata)


def group_by_split(annotations: Sequence[Annotation]) -> Dict[str, List[Annotation]]:
    grouped = {split: [] for split in OUTPUT_SPLITS}
    for annotation in annotations:
        grouped[annotation.output_split].append(annotation)
    return grouped


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("--prompt must not be empty")

    prepare_output_dir(source_dir, output_dir, args.overwrite)
    annotations = load_annotations(source_dir / "flickr_annotations_30k.csv")
    image_source = detect_image_source(source_dir, args.image_source)
    validate_images(annotations, image_source)

    grouped = group_by_split(annotations)
    split_stats: Dict[str, Dict[str, Any]] = {}
    with ImageMaterializer(image_source) as materializer:
        for split_name in OUTPUT_SPLITS:
            split_stats[split_name] = write_split(
                split_name=split_name,
                annotations=grouped[split_name],
                materializer=materializer,
                output_dir=output_dir,
                prompt=prompt,
                captions_per_image=args.captions_per_image,
                write_llava=not args.skip_llava_json,
            )

    write_metadata(
        output_dir=output_dir,
        source_dir=source_dir,
        image_source=image_source,
        prompt=prompt,
        captions_per_image=args.captions_per_image,
        write_llava=not args.skip_llava_json,
        split_stats=split_stats,
    )

    print("Finished Flickr30k conversion.")
    for split_name in OUTPUT_SPLITS:
        stats = split_stats[split_name]
        print(f"  {split_name}: {stats['images']} images -> {stats['samples']} samples")
    print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
