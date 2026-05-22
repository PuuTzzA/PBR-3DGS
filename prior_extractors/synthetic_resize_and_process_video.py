import os
import re
from PIL import Image

def process_video_ir_images(original_dir, video_ir_dir, output_dir):
    # Category mapping (ignoring 'rgb')
    categories = {
        'basecolor': 'albedo',
        'depth': 'depth',
        'metallic': 'metallic',
        'normal': 'normal',
        'roughness': 'roughness'
    }

    # Regex to match processed images (e.g., test_000_rgba.basecolor.png)
    # Group 1: prefix (test, train, val)
    # Group 2: index (000, 001...)
    # Group 3: category (basecolor, depth...)
    file_pattern = re.compile(r"^(test|train|val)_(\d+)_rgba\.([^.]+)\.png$")

    # The subfolders we expect in VIDEO_IR_DIR
    splits = ['test', 'train', 'val']

    for prefix in splits:
        split_dir = os.path.join(video_ir_dir, prefix)
        if not os.path.exists(split_dir):
            print(f"Skipping '{prefix}': Folder not found in {video_ir_dir}")
            continue

        print(f"\nProcessing '{prefix}' folder...")

        for file in os.listdir(split_dir):
            match = file_pattern.match(file)
            if not match:
                continue

            file_prefix = match.group(1)  # 'test', 'train', or 'val'
            index = match.group(2)  # '000', '001', etc.
            raw_cat = match.group(3)  # 'basecolor', 'depth', etc.

            # Skip '.rgb.png' or any unknown categories
            if raw_cat not in categories:
                continue

            out_cat = categories[raw_cat]

            # 1. Paths to images
            processed_file_path = os.path.join(split_dir, file)
            base_filename = f"{file_prefix}_{index}_rgba.png"
            base_file_path = os.path.join(original_dir, base_filename)

            if not os.path.exists(base_file_path):
                print(f"Warning: Original image missing for {file} -> Expected at: {base_file_path}")
                continue

            # 2. Extract size and alpha from original image
            with Image.open(base_file_path) as orig_img:
                orig_img = orig_img.convert("RGBA")
                target_size = orig_img.size
                alpha_channel = orig_img.getchannel('A')

            # 3. Process the video_ir image
            with Image.open(processed_file_path) as proc_img:
                proc_img = proc_img.convert("RGB")  # Remove broken artifacts/background

                try:
                    resample_method = Image.Resampling.BICUBIC
                except AttributeError:
                    resample_method = Image.BICUBIC  # Fallback for older Pillow

                # Resize and re-add clean alpha channel
                resized_proc_img = proc_img.resize(target_size, resample=resample_method)
                resized_proc_img.putalpha(alpha_channel)

                # 4. Construct output directory (e.g. output_dir/test/albedo_video/)
                out_folder_name = f"{out_cat}_video"
                out_dir = os.path.join(output_dir, file_prefix, out_folder_name)
                os.makedirs(out_dir, exist_ok=True)

                # 5. Output filename (e.g. albedo_000.png)
                out_filename = f"{out_cat}_{index}.png"
                out_filepath = os.path.join(out_dir, out_filename)

                resized_proc_img.save(out_filepath)
                print(f"Saved: {file_prefix}/{out_folder_name}/{out_filename}")

    print(f"\nFinished processing all video_ir images. Results saved in '{output_dir}'.")

# =====================================================================
# CONFIGURATION: Specify your directories here
# =====================================================================
# 1. Folder containing original images (test_000_rgba.png) for Alpha & Size
ORIGINAL_IMAGES_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\armadillo_images"

# 2. Folder containing the processed video_ir output (with test, train, val subfolders)
VIDEO_IR_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\armadillo_images\video_ir"

# 3. Output base folder where the new structure will be saved
OUTPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\armadillo_images_resized_video"

if __name__ == "__main__":
    process_video_ir_images(ORIGINAL_IMAGES_DIR, VIDEO_IR_DIR, OUTPUT_DIR)
