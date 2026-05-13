import os
import sys
import requests
import zipfile
from tqdm.auto import tqdm
import shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download

RESULT_DIR = "./data"


# 1. Zenodo Files
def download_with_progress(url, destination):
    # Use a realistic User-Agent to prevent Zenodo's WAF from blocking the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    response = requests.get(url, stream=True, headers=headers)

    # Immediately check for HTTP errors (e.g., 403 Forbidden, 404 Not Found)
    # This prevents the script from saving an HTML error page as a .zip file
    response.raise_for_status()

    # Check if we got the file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            disable=False,  # Force display
            leave=True
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)


def extract_zip_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        # Progress bar based on number of files inside the zip
        for file in tqdm(files, desc=f"Extracting {os.path.basename(zip_path)}", disable=False):
            z.extract(file, extract_to)


zenodo_links = {
    "armadillo.zip": "https://zenodo.org/records/7880113/files/armadillo.zip?download=1",
    "lego.zip": "https://zenodo.org/records/7880113/files/lego.zip?download=1"
}

os.makedirs(RESULT_DIR, exist_ok=True)

# 1. Zenodo Files
failed_downloads = []

for filename, url in zenodo_links.items():
    zip_filepath = os.path.join(RESULT_DIR, filename)

    # Get the folder name without '.zip' (e.g., 'lego' or 'armadillo')
    folder_name = filename.replace('.zip', '')
    extract_dir = os.path.join(RESULT_DIR, folder_name)

    # Skip if we already extracted this folder previously
    if os.path.exists(extract_dir):
        print(f"Skipping {filename}: Already extracted to {extract_dir}")
        continue

    # If the zip doesn't exist, try to download it automatically
    if not os.path.exists(zip_filepath):
        try:
            download_with_progress(url, zip_filepath)
        except Exception as e:
            # Record the failure and move on to check the next file
            failed_downloads.append((filename, url))
            continue

    # Extract the file into its specific subfolder (e.g., ./data/lego)
    os.makedirs(extract_dir, exist_ok=True)
    extract_zip_with_progress(zip_filepath, extract_dir)

    # Clean up the zip file after extraction to save space
    os.remove(zip_filepath)

# If any downloads failed, print them all out at once and exit
if failed_downloads:
    print("\n" + "=" * 60)
    print("[ERROR] The following files could not be downloaded automatically.")
    print(f"Please manually download them and place them in the '{RESULT_DIR}' folder:\n")

    for filename, url in failed_downloads:
        print(f"-> {filename}")
        print(f"   Download link: {url}\n")

    print("After placing the files, run this script again.")
    print("=" * 60)
    sys.exit(1)

# 2. Hugging Face Files
REPO_ID = "nvs-bench/mipnerf360"
folders = ["bicycle/*", "garden/*"]

print(f"\nStarting Hugging Face download...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=folders,
    local_dir=RESULT_DIR,
)

print(f"Done Downloading! Files are in {RESULT_DIR}")


# Move images into correct folders
def organize_datasets():
    # List of folders to process
    hf_folders = ["bicycle", "garden"]
    zenodo_folders = ["lego", "armadillo"]

    for folder_name in (hf_folders + zenodo_folders):
        src_path = os.path.join(RESULT_DIR, folder_name)

        # Skip if the source folder doesn't exist
        if not os.path.exists(src_path):
            print(f"Skipping {folder_name}: Folder not found.")
            continue

        # Create the new destination folder (e.g., ./data/bicycle_images)
        dest_path = os.path.join(RESULT_DIR, f"{folder_name}_images")
        os.makedirs(dest_path, exist_ok=True)

        print(f"Processing {folder_name}...")

        # --- Strategy A: Hugging Face Structure (bicycle/garden) ---
        if folder_name in hf_folders:
            images_subdir = os.path.join(src_path, "images")
            if os.path.exists(images_subdir):
                files = sorted(os.listdir(images_subdir))
                for i, f in enumerate(tqdm(files, desc=f"Copying and renaming {folder_name} images")):
                    # Get file extension
                    ext = os.path.splitext(f)[1]
                    # Create new filename: foldername_000.ext
                    new_filename = f"{folder_name}_{i:03d}{ext}"
                    shutil.copy2(
                        os.path.join(images_subdir, f),
                        os.path.join(dest_path, new_filename)
                    )
            else:
                print(f"Warning: Could not find 'images' subfolder in {src_path}")

        # --- Strategy B: Zenodo Structure (lego/armadillo) ---
        elif folder_name in zenodo_folders:
            # We look for any 'rgba.png' inside subdirectories
            rgba_files = []
            for root, dirs, files in os.walk(src_path):
                if "rgba.png" in files:
                    rgba_files.append(os.path.join(root, "rgba.png"))

            for file_path in tqdm(rgba_files, desc=f"Extracting {folder_name} rgba.pngs"):
                # Get the name of the parent folder (e.g., 'test_001')
                parent_folder = os.path.basename(os.path.dirname(file_path))

                # Create a unique name: test_001_rgba.png
                new_filename = f"{parent_folder}_rgba.png"

                shutil.copy2(
                    file_path,
                    os.path.join(dest_path, new_filename)
                )

    print("\nOrganization complete!")
    print(f"Your processed folders are in: {os.path.abspath(RESULT_DIR)}")


organize_datasets()
