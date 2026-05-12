# download and install all dependencies for cosmos https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer
# for steps to run it in google colab: https://colab.research.google.com/drive/17KGk4DGgmPixbzqMdbtshNsAe6A-sQc1?usp=sharing
# it runs on google colab, but on the free tier there is not enough VRAM

import os
import subprocess

DATASETS = ["bicycle", "garden", "lego", "armadillo"]
BASE_INPUT_DIR = "./data"
BASE_OUTPUT_DIR = "./data_results"

# Cosmos specific paths
# Ensure you are running this script from the root of the Cosmos repository
CHECKPOINT_DIR = "checkpoints"
MODEL_VARIANT = "Diffusion_Renderer_Inverse_Cosmos_7B"
INFERENCE_SCRIPT = "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py"

def run_inverse_rendering():
    # 1. Prepare Environment Variables
    # Mimics: CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd)
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX", "")
    
    if not conda_prefix:
        print("Warning: CONDA_PREFIX not found. Ensure you are in a conda environment.")
    
    env["CUDA_HOME"] = conda_prefix
    env["PYTHONPATH"] = os.getcwd()

    for dataset in DATASETS:
        input_folder = os.path.join(BASE_INPUT_DIR, f"{dataset}_images")
        output_folder = os.path.join(BASE_OUTPUT_DIR, dataset)
        
        if not os.path.exists(input_folder):
            print(f"Skipping {dataset}: Input folder {input_folder} does not exist.")
            continue

        print(f"\n" + "="*50)
        print(f"STARTING INVERSE RENDERING: {dataset}")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print("="*50 + "\n")

        # Construct the command
        cmd = [
            "python", INFERENCE_SCRIPT,
            "--checkpoint_dir", CHECKPOINT_DIR,
            "--diffusion_transformer_dir", MODEL_VARIANT,
            "--dataset_path", input_folder,
            "--num_video_frames", "1",
            "--group_mode", "webdataset",
            "--video_save_folder", output_folder,
            "--save_video", "False"
        ]

        # Optional: If you encounter OOM (Out of Memory), uncomment these lines:
        # cmd.append("--offload_diffusion_transformer")
        # cmd.append("--offload_tokenizer")

        try:
            # Run the command and stream output to console
            subprocess.run(cmd, env=env, check=True)
            print(f"\nSuccessfully processed {dataset}")
        except subprocess.CalledProcessError as e:
            print(f"\nError occurred while processing {dataset}:")
            print(e)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            return

    print("\nAll datasets processed!")

if __name__ == "__main__":
    run_inverse_rendering()