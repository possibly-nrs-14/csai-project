import os
import subprocess
from pathlib import Path

def main():
    base_out_dir = Path("brain_vis/vjepa")
    sample_fractions = [1, 2, 4, 8, 10]
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "mean"]
    
    # Base pattern for the decoding roots
    algonauts_base = Path("/scratch/arihantr/CSAI/algonauts_outputs")
    
    for pct in sample_fractions:
        # Create output directory for this fraction
        out_dir = base_out_dir / str(pct)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to decoding results for this fraction
        decoding_root = algonauts_base / f"s1_vjepa_decode_uniform{pct}pct_oomfixed" / "decoding"
        
        for subj in subjects:
            out_file = out_dir / f"{subj.replace('-', '')}.png"
            
            print(f"Generating visualization for fraction {pct}%, subject {subj}...")
            
            # Construct the command
            cmd = [
                "python", "visualize_brain.py",
                "--decoding_root", str(decoding_root),
                "--subject", subj,
                "--output", str(out_file)
            ]
            
            # Run the command
            try:
                subprocess.run(cmd, check=True)
                print(f"Successfully saved to {out_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error generating plot for {subj} at {pct}%: {e}\n")

if __name__ == "__main__":
    main()
