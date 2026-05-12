import os
import requests
import zipfile
from tqdm.auto import tqdm  
import shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download

RESULT_DIR = "./data"

def download_with_progress(url, destination):
    # Added headers because some servers refuse to give 'content-length' to scripts
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    
    # Check if we got the file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 

    with open(destination, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        disable=False, # Force display
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
for filename, url in zenodo_links.items():
    zip_filepath = os.path.join(RESULT_DIR, filename)
    download_with_progress(url, zip_filepath)
    extract_zip_with_progress(zip_filepath, RESULT_DIR)
    os.remove(zip_filepath)

# 2. Hugging Face Files
REPO_ID = "nvs-bench/mipnerf360"
folders = ["bicycle/*", "garden/*"]

print(f"\nStarting Hugging Face download...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=folders,
    local_dir=RESULT_DIR,
    local_dir_use_symlinks=False,
    tqdm_class=tqdm # Explicitly tell HF to use our tqdm
)

print(f"Done! Files are in {RESULT_DIR}")

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
                files = os.listdir(images_subdir)
                for f in tqdm(files, desc=f"Copying {folder_name} images"):
                    shutil.copy2(
                        os.path.join(images_subdir, f), 
                        os.path.join(dest_path, f)
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

if __name__ == "__main__":
    organize_datasets()