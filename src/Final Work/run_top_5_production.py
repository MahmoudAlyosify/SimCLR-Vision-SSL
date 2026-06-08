import os
import subprocess
import shutil
import sys
import time

def setup_and_run(exp_id, folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    target_dir = os.path.join(base_dir, folder_name)
    
    print("\n" + "="*80)
    print(f" Preparing and Running Experiment {exp_id} in {folder_name} (200 Epochs)")
    print("="*80)
    
    # 1. Copy src files and folders to the target folder
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(target_dir, file_name)
        if os.path.isdir(src_file):
            if os.path.exists(dst_file):
                shutil.rmtree(dst_file)
            shutil.copytree(src_file, dst_file)
            print(f"Copied directory {file_name} to {folder_name}")
        elif file_name.endswith(".py"):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {file_name} to {folder_name}")
            
    # 2. Run the training script from within the target folder
    # We use the same python executable that is running this script
    cmd = [
        sys.executable, "train_master.py",
        "--epochs", "200",
        "--batch_size", "512",
        "--backbone", "resnet50",
        "--no_compile",
        "--exp_id", str(exp_id),
        "--output_dir", "./outputs"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    
    try:
        # Run subprocess and stream output
        result = subprocess.run(cmd, cwd=target_dir)
        if result.returncode != 0:
            print(f"\nExperiment {exp_id} failed with error code {result.returncode}")
        else:
            print(f"\nExperiment {exp_id} completed successfully!")
    except Exception as e:
        print(f"\nFailed to run Experiment {exp_id}: {e}")

if __name__ == "__main__":
    experiments = [
        (36, "v2_pure_rotation"),
        (35, "v3_weak_baseline"),
        (9,  "v4_crop_blur"),
        (13, "v5_crop_flip_blur"),
        (10, "v6_crop_cutout")
    ]
    
    start_time = time.time()
    
    for exp_id, folder in experiments:
        setup_and_run(exp_id, folder)
        
    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*80)
    print(f" ALL 5 EXPERIMENTS COMPLETED! Total Time: {total_time:.2f} hours")
    print("="*80)
