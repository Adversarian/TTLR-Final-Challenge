"""Utility to download and extract Torob dataset from Google Drive."""

import argparse
import pathlib
import shutil
import tarfile

import gdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-id",
        type=str,
        required=True,
        help="Google Drive file ID for the dataset archive.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("torob-turbo-stage2.tar.gz"),
        help="Destination path for the downloaded archive.",
    )
    parser.add_argument(
        "--extract-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to extract the archive contents into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archive file and extracted directory if present.",
    )
    return parser.parse_args()


def download_archive(
    file_id: str, output_path: pathlib.Path, overwrite: bool
) -> pathlib.Path:
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(output_path), quiet=False)
    return output_path


def extract_archive(
    archive_path: pathlib.Path, extract_dir: pathlib.Path, overwrite: bool
) -> None:
    if extract_dir.exists():
        if not overwrite:
            return
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def main() -> None:
    args = parse_args()
    archive_path = download_archive(args.drive_id, args.output, args.overwrite)

    if args.extract_dir is not None:
        extract_dir = args.extract_dir
        extract_archive(archive_path, extract_dir, args.overwrite)
        print(f"Extracted dataset to {extract_dir}")
    else:
        print(f"Dataset downloaded to {archive_path}")


if __name__ == "__main__":
    main()
