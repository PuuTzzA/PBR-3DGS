import os
import re
from PIL import Image

def process_images(input_path, output_path):
    # The categories of images we want to process (ignoring 'rgb')
    # and mapping 'basecolor' to 'albedo'
    categories = {
        'basecolor': 'albedo',
        'depth': 'depth',
        'metallic': 'metallic',
        'normal': 'normal',
        'roughness': 'roughness'
    }

    # Regex to match the base input images (e.g., test_000_rgba.png)
    # Group 1: prefix (test, train, val)
    # Group 2: index (000, 001, ...)
    base_pattern = re.compile(r"^(test|train|val)_(\d+)_rgba\.png$")

    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    # Gather all base files first
    base_files = [f for f in os.listdir(input_path) if base_pattern.match(f)]

    if not base_files:
        print("No input images found matching the pattern '{test|train|val}_{index}_rgba.png'.")
        return

    print(f"Found {len(base_files)} base images. Starting processing...")

    for base_file in base_files:
        match = base_pattern.match(base_file)
        prefix = match.group(1)  # test, train, or val
        index = match.group(2)  # 000, 001, etc.

        base_file_path = os.path.join(input_path, base_file)

        # Load the original image and extract the exact alpha channel to preserve it
        with Image.open(base_file_path) as orig_img:
            orig_img = orig_img.convert("RGBA")
            target_size = orig_img.size
            alpha_channel = orig_img.getchannel('A')

        # Process each valid output category
        for raw_cat, out_cat in categories.items():
            processed_filename = f"{prefix}_{index}_rgba.{raw_cat}.png"
            processed_file_path = os.path.join(input_path, processed_filename)

            if not os.path.exists(processed_file_path):
                print(f"Warning: Processed file not found -> {processed_filename}")
                continue

            # Load processed image, resize it, and apply the original alpha channel
            with Image.open(processed_file_path) as proc_img:
                # Convert to RGB just in case, removing any broken background alpha
                proc_img = proc_img.convert("RGB")

                # Resize using Bicubic interpolation
                try:
                    resample_method = Image.Resampling.BICUBIC  # Pillow 9.0.0+
                except AttributeError:
                    resample_method = Image.BICUBIC  # Older Pillow versions

                resized_proc_img = proc_img.resize(target_size, resample=resample_method)

                # Re-add the clean alpha channel from the original input image
                resized_proc_img.putalpha(alpha_channel)

                # Construct output directory path: e.g., output_path/test/albedo/
                out_dir = os.path.join(output_path, prefix, out_cat)
                os.makedirs(out_dir, exist_ok=True)

                # Construct the final output filename: e.g., albedo_000.png
                out_filename = f"{out_cat}_{index}.png"
                out_filepath = os.path.join(out_dir, out_filename)

                resized_proc_img.save(out_filepath)

        print(f"Processed: {base_file}")

    print(f"\nFinished processing all images. Results saved in '{output_path}'.")

# =====================================================================
# CONFIGURATION: Specify your input and output directories here
# Use 'r' before the string on Windows to handle backslashes correctly
# =====================================================================
INPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\bicycle_images"  # Folder containing test_000, transforms_test.json, etc.
OUTPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\bicycle_images_resized"  # Folder where the new structure will be saved


if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)