import argparse
import array
import shutil
import struct
import sys
from pathlib import Path


def _read_images_header(path):
    with path.open("rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))

    if magic != 2051:
        raise ValueError(f"Invalid MNIST images file: {path}")

    return count, rows, cols


def _read_labels_header(path):
    with path.open("rb") as file:
        magic, count = struct.unpack(">II", file.read(8))

    if magic != 2049:
        raise ValueError(f"Invalid MNIST labels file: {path}")

    return count


def _export_images_float32(images_path, output_path, count, rows, cols):
    image_size = rows * cols
    chunk_images = 512
    chunk_bytes = chunk_images * image_size

    with images_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(16)
        remaining = count * image_size

        while remaining > 0:
            raw = src.read(min(chunk_bytes, remaining))
            if not raw:
                raise ValueError(f"Unexpected end of image file: {images_path}")

            values = array.array("f", (pixel / 255.0 for pixel in raw))
            if sys.byteorder != "little":
                values.byteswap()

            values.tofile(dst)
            remaining -= len(raw)


def _export_labels_uint8(labels_path, output_path, count):
    with labels_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(8)
        shutil.copyfileobj(src, dst)

    if output_path.stat().st_size != count:
        raise ValueError(f"Unexpected label count in {labels_path}")


def export_split(source_dir, output_dir, split, images_name, labels_name):
    images_path = source_dir / images_name
    labels_path = source_dir / labels_name

    count, rows, cols = _read_images_header(images_path)
    label_count = _read_labels_header(labels_path)

    if count != label_count:
        raise ValueError(f"Image/label count mismatch for {split}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images_output = output_dir / f"{split}_images_float32.bin"
    labels_output = output_dir / f"{split}_labels_uint8.bin"

    _export_images_float32(images_path, images_output, count, rows, cols)
    _export_labels_uint8(labels_path, labels_output, count)

    print(f"{split}: {count} images, shape=({count}, 1, {rows}, {cols})")
    print(f"  images -> {images_output}")
    print(f"  labels -> {labels_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Export MNIST IDX files to flat binary files for a C++ data loader."
    )
    parser.add_argument("--source", default="datasets/mnist", help="Folder containing MNIST IDX files.")
    parser.add_argument("--out", default="datasets/mnist/flat", help="Output folder for flat binary files.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.out)

    export_split(
        source_dir,
        output_dir,
        "train",
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
    )
    export_split(
        source_dir,
        output_dir,
        "test",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
    )


if __name__ == "__main__":
    main()
