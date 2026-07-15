import os
import glob
import imageio

def make_video_loop(image_dir, output_path, is_gt=False, fps=30):
    pattern = "test_rgba_rgba_*_gt.png" if is_gt else "test_rgba_rgba_*_rendered.png"
    images = sorted(glob.glob(os.path.join(image_dir, pattern)))
    
    if not images:
        print(f"Skipping: No frames found in {image_dir} for pattern {pattern}")
        return False
        
    print(f"Reading {len(images)} frames from {image_dir}...")
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
        for path in images:
            frame = imageio.imread(path)
            writer.append_data(frame)
        writer.close()
        print(f"Successfully generated: {output_path}")
        return True
    except Exception as e:
        print(f"Error compiling video {output_path}: {e}")
        return False

def main():
    base_dir = "outputs/poster_results_with_evaluation"
    target_dir = "docs/static/videos"
    os.makedirs(target_dir, exist_ok=True)
    
    scenes = ["lego", "hotdog"]
    runs = {
        "lego": {
            "baseline": "lego_baseline_no_prior",
            "gt_prior": "lego_gt_zncc_zncc_neu",
            "diff_prior": "lego_diff_zncc_zncc"
        },
        "hotdog": {
            "baseline": "hotdog_baseline_no_prior",
            "gt_prior": "hotdog_gt_zncc_zncc_neu",
            "diff_prior": "hotdog_diff_zncc_zncc"
        }
    }
    environments = ["fireplace", "night", "snow"]
    
    for scene in scenes:
        scene_runs = runs[scene]
        
        # Compile Albedo Video Loops
        print(f"\n--- Processing Scene: {scene}, Albedo ---")
        make_video_loop(
            image_dir=os.path.join(base_dir, scene, scene_runs["baseline"], "evaluation_albedo"),
            output_path=os.path.join(target_dir, f"{scene}_baseline_albedo.mp4"),
            is_gt=False
        )
        make_video_loop(
            image_dir=os.path.join(base_dir, scene, scene_runs["gt_prior"], "evaluation_albedo"),
            output_path=os.path.join(target_dir, f"{scene}_gt_prior_albedo.mp4"),
            is_gt=False
        )
        make_video_loop(
            image_dir=os.path.join(base_dir, scene, scene_runs["diff_prior"], "evaluation_albedo"),
            output_path=os.path.join(target_dir, f"{scene}_diff_prior_albedo.mp4"),
            is_gt=False
        )
        make_video_loop(
            image_dir=os.path.join(base_dir, scene, scene_runs["baseline"], "evaluation_albedo"),
            output_path=os.path.join(target_dir, f"{scene}_gt_albedo.mp4"),
            is_gt=True
        )

        # Compile Relighting Video Loops
        for env in environments:
            print(f"\n--- Processing Scene: {scene}, Env: {env} ---")
            
            # 1. Baseline
            make_video_loop(
                image_dir=os.path.join(base_dir, scene, scene_runs["baseline"], f"evaluation_{env}"),
                output_path=os.path.join(target_dir, f"{scene}_baseline_{env}.mp4"),
                is_gt=False
            )
            
            # 2. GT Prior
            make_video_loop(
                image_dir=os.path.join(base_dir, scene, scene_runs["gt_prior"], f"evaluation_{env}"),
                output_path=os.path.join(target_dir, f"{scene}_gt_prior_{env}.mp4"),
                is_gt=False
            )
            
            # 3. Diffusion Prior
            make_video_loop(
                image_dir=os.path.join(base_dir, scene, scene_runs["diff_prior"], f"evaluation_{env}"),
                output_path=os.path.join(target_dir, f"{scene}_diff_prior_{env}.mp4"),
                is_gt=False
            )
            
            # 4. Ground Truth
            make_video_loop(
                image_dir=os.path.join(base_dir, scene, scene_runs["baseline"], f"evaluation_{env}"),
                output_path=os.path.join(target_dir, f"{scene}_gt_{env}.mp4"),
                is_gt=True
            )

if __name__ == "__main__":
    main()
