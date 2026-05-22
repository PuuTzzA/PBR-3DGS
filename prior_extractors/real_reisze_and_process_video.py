import os
import re
from PIL import Image

# =====================================================================
# CONFIGURATION: Specify your directories here
# =====================================================================
# Main folder containing original images (bicycle_000.JPG) for Alpha & Size
INPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\garden_images"

# The specific subfolder containing the video_ir images
VIDEO_IR_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\garden_images\video_ir\garden"

# Output base folder
OUTPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\processed\garden_images_resized"


def _resize_and_save(proc_path, target_size, alpha_channel, output_base, folder_name, cat_prefix, index):
    """Helper function to resize a single image and save it to the target directory."""
    with Image.open(proc_path) as proc_img:
        # Convert to RGB to strip out any problematic background artifacts
        proc_img = proc_img.convert("RGB")

        # Resize using Bicubic interpolation
        try:
            resample_method = Image.Resampling.BICUBIC
        except AttributeError:
            resample_method = Image.BICUBIC

        resized_img = proc_img.resize(target_size, resample=resample_method)

        # If the original image had an alpha channel, paste it back in
        if alpha_channel is not None:
            resized_img.putalpha(alpha_channel)

        # Ensure the output directory exists
        out_dir = os.path.join(output_base, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        # Save the file (e.g., albedo_000.png)
        out_name = f"{cat_prefix}_{index}.png"
        resized_img.save(os.path.join(out_dir, out_name))


def process_video_ir_only():
    categories = {
        'basecolor': 'albedo',
        'depth': 'depth',
        'metallic': 'metallic',
        'normal': 'normal',
        'roughness': 'roughness'
    }

    if not os.path.exists(VIDEO_IR_DIR):
        print(f"Error: Video IR directory '{VIDEO_IR_DIR}' does not exist.")
        return

    # 1. Identify all unique base names in the video_ir/bicycle directory
    base_names = set()
    for f in os.listdir(VIDEO_IR_DIR):
        if f.endswith('.basecolor.png'):
            base_names.add(f.replace('.basecolor.png', ''))

    if not base_names:
        print("No processed images found matching the pattern '*.basecolor.png' in the video_ir folder.")
        return

    print(f"Found {len(base_names)} sets of images in video_ir. Starting processing...\n")

    for base_name in sorted(base_names):
        # Extract the index number from the base name ('bicycle_000' -> '000')
        match = re.search(r'_(\d+)(?:_rgba)?$', base_name)
        index = match.group(1) if match else base_name

        # 2. Find the original image in the MAIN directory to get dimensions & alpha
        orig_path = None
        for ext in ['.JPG', '.jpg', '.png', '.jpeg']:
            test_path = os.path.join(INPUT_DIR, base_name + ext)
            if os.path.exists(test_path):
                orig_path = test_path
                break

        if not orig_path:
            print(f"Warning: Could not find original image for '{base_name}' in {INPUT_DIR}. Skipping.")
            continue

        # 3. Read the original image to extract target size and optional Alpha channel
        with Image.open(orig_path) as orig_img:
            target_size = orig_img.size

            # Check if image actually has transparency (JPGs do not, RGBA PNGs do)
            has_alpha = orig_img.mode in ('RGBA', 'LA') or (orig_img.mode == 'P' and 'transparency' in orig_img.info)
            if has_alpha:
                orig_img = orig_img.convert("RGBA")
                alpha_channel = orig_img.getchannel('A')
            else:
                alpha_channel = None

        # 4. Process ONLY the video_ir files
        for raw_cat, out_cat in categories.items():
            proc_path_vid = os.path.join(VIDEO_IR_DIR, f"{base_name}.{raw_cat}.png")

            if os.path.exists(proc_path_vid):
                # Save into `_video` suffixed folders like `albedo_video`
                folder_name_vid = f"{out_cat}_video"
                _resize_and_save(proc_path_vid, target_size, alpha_channel, OUTPUT_DIR, folder_name_vid, out_cat, index)

        print(f"Processed: {base_name} (Alpha Mask Re-added: {'Yes' if alpha_channel else 'No'})")

    print(f"\nFinished! Cleaned video_ir data saved to: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    process_video_ir_only()
