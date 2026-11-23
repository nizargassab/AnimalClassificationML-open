import argparse
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory

from config import BASE_DIR, SEED

EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def split_dataset(source, output):
    if not source.is_dir():
        raise ValueError(f"Dataset directory does not exist: {source}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Choose a new output directory: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("The output directory must be outside the source dataset.")

    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError("Create at least two class directories in the dataset.")

    random = Random(SEED)
    class_splits = {}
    for directory in classes:
        images = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if len(images) < 10:
            raise ValueError(f"Class '{directory.name}' needs at least 10 images.")
        random.shuffle(images)
        held_out = len(images) // 10
        class_splits[directory.name] = {
            "validation": images[:held_out],
            "test": images[held_out:2 * held_out],
            "train": images[2 * held_out:],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dataset-split-", dir=output.parent) as tmp:
        split_root = Path(tmp) / "dataset"
        for name, splits in class_splits.items():
            for split, images in splits.items():
                destination = split_root / split / name
                destination.mkdir(parents=True)
                for image in images:
                    copyfile(image, destination / image.name)
        split_root.rename(output)


def main():
    parser = argparse.ArgumentParser(description="Split class folders into train, validation and test sets.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    try:
        split_dataset(args.dataset, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Dataset splits saved to {args.output}")


if __name__ == "__main__":
    main()
