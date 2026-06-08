import os
import subprocess
import shutil
import sys
import time

def setup_and_run(exp_id, folder_name, base_output_dir):
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    target_dir = os.path.join(base_output_dir, folder_name)
    
    print("\n" + "="*80)
    print(f" Preparing and Running Jittered Experiment {exp_id} in {folder_name} (200 Epochs)")
    print("="*80)
    
    os.makedirs(target_dir, exist_ok=True)
    
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
        result = subprocess.run(cmd, cwd=target_dir)
        if result.returncode != 0:
            print(f"\nExperiment {exp_id} failed with error code {result.returncode}")
        else:
            print(f"\nExperiment {exp_id} completed successfully!")
    except Exception as e:
        print(f"\nFailed to run Experiment {exp_id}: {e}")

if __name__ == "__main__":
    experiments = [
        (38, "v9_rotation_jitter_exp38"),
        (39, "v10_weakbaseline_jitter_exp39"),
        (40, "v11_blur_jitter_exp40"),
        (41, "v12_flipblur_jitter_exp41"),
        (42, "v13_cutout_jitter_exp42")
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.join(base_dir, "All Experiment SimCLR 18 May 2026")
    os.makedirs(pipeline_dir, exist_ok=True)
    
    start_time = time.time()
    
    for exp_id, folder in experiments:
        setup_and_run(exp_id, folder, pipeline_dir)
        
    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*80)
    print(f" ALL 5 JITTERED ABLATIONS COMPLETED! Total Time: {total_time:.2f} hours")
    print("="*80)
    
    print("\n[+] Starting Automated Linear Probe Evaluation and Plotting...")
    eval_script = os.path.join(base_dir, "evaluate_and_plot_beast.py")
    subprocess.run([sys.executable, eval_script, pipeline_dir], cwd=base_dir)
    print("\n[+] Pipeline fully completed. Please check the 'All Experiment SimCLR 18 May 2026/Results' folder for the final outputs!")
