import json
import shutil
from pathlib import Path

# =====================================================================

def reorganize_dataset(input_dir, output_dir):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # Ensure the output directory exists
    out_path.mkdir(parents=True, exist_ok=True)

    splits = ["test", "train", "val"]

    for split in splits:
        print(f"\n--- Processing {split} split ---")

        # 1. Create the new directory structure in the OUTPUT folder
        split_dir = out_path / split
        (split_dir / "rgba").mkdir(parents=True, exist_ok=True)
        (split_dir / "normal_gt").mkdir(parents=True, exist_ok=True)
        (split_dir / "albedo_gt").mkdir(parents=True, exist_ok=True)
        (split_dir / "other_gt").mkdir(parents=True, exist_ok=True)

        # 2. Find all matching folders in the INPUT folder (e.g., test_000, test_001)
        pattern = f"{split}_[0-9][0-9][0-9]"
        old_dirs = sorted(in_path.glob(pattern))

        if not old_dirs:
            print(f"No folders found for '{split}'. Skipping...")
            continue

        for dir_path in old_dirs:
            # Extract the ID string (e.g., '000' from 'test_000')
            id_str = dir_path.name.split("_")[-1]

            # Copy and rename files
            for file_path in dir_path.iterdir():
                if not file_path.is_file():
                    continue

                ext = file_path.suffix
                stem = file_path.stem

                if file_path.name == "rgba.png":
                    new_name = f"rgba_{id_str}{ext}"
                    dest = split_dir / "rgba" / new_name
                elif file_path.name == "normal.png":
                    new_name = f"normal_{id_str}{ext}"
                    dest = split_dir / "normal_gt" / new_name
                elif file_path.name == "albedo.png":
                    new_name = f"albedo_{id_str}{ext}"
                    dest = split_dir / "albedo_gt" / new_name
                else:
                    # Rename all other files to prevent overwriting in the shared folder
                    new_name = f"{stem}_{id_str}{ext}"
                    dest = split_dir / "other_gt" / new_name

                # Copy the file to the new destination (preserves the original)
                shutil.copy2(str(file_path), str(dest))

            print(f"Processed {dir_path.name}")

        # 3. Update the transforms JSON file
        in_json = in_path / f"transforms_{split}.json"
        out_json = out_path / f"transforms_{split}.json"

        if in_json.exists():
            with open(in_json, "r") as f:
                data = json.load(f)

            if "frames" in data:
                for frame in data["frames"]:
                    old_path = frame["file_path"]

                    # Extract the old folder name from the path (e.g., "./test_000/rgba")
                    parts = old_path.split("/")
                    if len(parts) >= 2:
                        folder_name = parts[1]
                        frame_id_str = folder_name.split("_")[-1]
                        frame_id_int = int(frame_id_str)

                        # Set the new path format (e.g., "./test/rgba/rgba_000")
                        frame["file_path"] = f"./{split}/rgba/rgba_{frame_id_str}"

                        # Add the file_id attribute
                        frame["file_id"] = frame_id_int

            # Save the modified JSON to the OUTPUT directory
            with open(out_json, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Successfully created updated JSON: {out_json}")
        else:
            print(f"Warning: {in_json} not found.")

# =====================================================================
# CONFIGURATION: Specify your input and output directories here
# Use 'r' before the string on Windows to handle backslashes correctly
# =====================================================================
INPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\armadillo"  # Folder containing test_000, transforms_test.json, etc.
OUTPUT_DIR = r"C:\__TUM__\M.Sc._Informatics_Games_Engineering\TUM_2_Semester_(SOSE_2026)\Advanced_Deep_Learning_4_Visual_Computing\PBR-3DGS\prior_extractors\data\data_clean\armadillo"  # Folder where the new structure will be saved

if __name__ == "__main__":
    reorganize_dataset(INPUT_DIR, OUTPUT_DIR)
    print("\nDataset successfully copied and reorganized!")
