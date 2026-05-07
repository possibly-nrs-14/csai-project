import os
import re
import subprocess

# --- CONFIGURATION ---
# The name of your original python file
ORIGINAL_SCRIPT = "ijepa_collect_embeddings.py" 

# The base path for your output (the script will append _8f, _16f, etc.)
BASE_OUTPUT_ROOT = "/scratch/arihantr/CSAI/algonauts_outputs/s1_ijepa_10"

# Define the frame counts and their corresponding OOM schedules
EXPERIMENTS = [
    {"frames": 8,  "schedule": "[8, 6, 4, 2]"},
    {"frames": 16, "schedule": "[16, 12, 8, 4, 1]"},
    {"frames": 32, "schedule": "[32, 24, 16, 8, 4]"},
    {"frames": 64, "schedule": "[32, 24, 16, 8, 4]"}, # Warning: Might need smaller batch_size for 32!
    {"frames": 128, "schedule": "[32, 24, 16, 8, 4]"}, # Warning: Might need smaller batch_size for 32!
]
# ---------------------

def main():
    # 1. Read the original script
    if not os.path.exists(ORIGINAL_SCRIPT):
        print(f"Error: Could not find {ORIGINAL_SCRIPT}")
        return

    with open(ORIGINAL_SCRIPT, "r") as f:
        original_code = f.read()

    # 2. Loop through each experiment setup
    for exp in EXPERIMENTS:
        frames = exp["frames"]
        schedule = exp["schedule"]
        out_dir = f"{BASE_OUTPUT_ROOT}_{frames}f"
        
        print(f"\n{'='*50}")
        print(f"🚀 PREPARING RUN: {frames} FRAMES PER CLIP")
        print(f"Output Directory: {out_dir}")
        print(f"{'='*50}")
        
        # 3. Use Regex to find and replace the dictionary values
        # Replaces "ijepa_frames_per_clip": <number>,
        new_code = re.sub(
            r'"ijepa_frames_per_clip":\s*\d+,', 
            f'"ijepa_frames_per_clip": {frames},', 
            original_code
        )
        
        # Replaces "oom_frame_retry_schedule": [<numbers>],
        new_code = re.sub(
            r'"oom_frame_retry_schedule":\s*\[.*?\],', 
            f'"oom_frame_retry_schedule": {schedule},', 
            new_code
        )
        
        # Replaces "output_root": "<path>",
        new_code = re.sub(
            r'"output_root":\s*".*?",', 
            f'"output_root": "{out_dir}",', 
            new_code
        )
        
        # 4. Save to a temporary script
        temp_script_name = f"temp_run_{frames}f.py"
        with open(temp_script_name, "w") as f:
            f.write(new_code)
            
        # 5. Execute the temporary script
        try:
            print(f"Running {temp_script_name}...")
            # subprocess.run waits for the script to finish before continuing the loop
            subprocess.run(["python", temp_script_name], check=True)
            print(f"✅ Successfully finished {frames} frame extraction.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred during the {frames} frame extraction.")
            print(e)
            break # Stop running future experiments if one crashes
        finally:
            # Clean up the temporary file so it doesn't clutter your directory
            if os.path.exists(temp_script_name):
                os.remove(temp_script_name)

if __name__ == "__main__":
    main()