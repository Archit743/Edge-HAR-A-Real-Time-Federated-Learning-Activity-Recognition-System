import zipfile
from pathlib import Path

import requests


def download_uci_har():
    """Downloads and extracts the UCI HAR dataset."""
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

    # Base data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "dataset.zip"
    extract_test_path = data_dir / "UCI HAR Dataset"

    if extract_test_path.exists():
        print("UCI HAR dataset already exists. Skipping download.")
        return

    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # The UCI HAR dataset actually comes as a zip inside a zip
        inner_zip = data_dir / "UCI HAR Dataset.zip"
        if inner_zip.exists():
            print("Extracting inner zip...")
            with zipfile.ZipFile(inner_zip, "r") as inner_ref:
                inner_ref.extractall(data_dir)
            inner_zip.unlink()  # Clean up the inner zip file
            
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to download or extract dataset: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up zip file.")

if __name__ == "__main__":
    download_uci_har()
