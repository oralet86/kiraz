#!/usr/bin/env python3
"""
Download and extract datasets from Google Drive.
Usage: python download_datasets.py
"""

import os
import sys
import zipfile
from pathlib import Path
import gdown


def load_env():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = Path(".env")

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    else:
        print("No .env file found.")
        print(
            "Create .env file with: GOOGLE_DRIVE_DATASET_URL=https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
        )
        return None

    return env_vars


def download_and_extract():
    """Download and extract datasets from Google Drive."""
    # Load environment variables
    env_vars = load_env()
    if not env_vars:
        return False

    # Get Google Drive URL
    url = env_vars.get("GOOGLE_DRIVE_DATASET_URL")
    if not url:
        print("GOOGLE_DRIVE_DATASET_URL not found in .env file")
        return False

    print(f"Downloading datasets from: {url}")

    # Download the file
    try:
        output = "datasets.zip"
        gdown.download(url, output, quiet=False, fuzzy=True)

        if not os.path.exists(output):
            print("Failed to download datasets.zip")
            return False

        print(f"Downloaded {output} ({os.path.getsize(output) / 1024**2:.1f} MB)")

    except Exception as e:
        print(f"Download failed: {e}")
        return False

    # Extract the zip file
    try:
        print("Extracting datasets...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            print(f"Archive contains {len(file_list)} files")

            # Check if there's a single top-level directory
            top_dirs = set()
            for path in file_list:
                if "/" in path:
                    top_dir = path.split("/")[0]
                    if not top_dir.startswith("."):
                        top_dirs.add(top_dir)

            # Extract all
            zip_ref.extractall(".")

            # If there's a single top-level directory that's not "datasets", rename it
            if len(top_dirs) == 1:
                old_name = list(top_dirs)[0]
                if old_name != "datasets":
                    if Path(old_name).exists():
                        print(f"Renaming '{old_name}' to 'datasets'")
                        Path(old_name).rename("datasets")

        print("Extraction successful")

        # Clean up zip file
        os.remove(output)
        print("Removed datasets.zip")

        return True

    except zipfile.BadZipFile as e:
        print(f"Invalid zip file: {e}")
        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def main():
    """Main function."""
    print("=== Dataset Download Script ===")

    success = download_and_extract()

    if success:
        print("✅ Dataset setup completed successfully!")

        # Show what was extracted
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            print(f"\n📁 Contents of {datasets_dir}:")
            for item in sorted(datasets_dir.iterdir()):
                if item.is_dir():
                    file_count = len(list(item.rglob("*")))
                    print(f"  📂 {item.name}/ ({file_count} files)")
                else:
                    size = item.stat().st_size / 1024**2
                    print(f"  📄 {item.name} ({size:.1f} MB)")

    else:
        print("❌ Dataset setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
