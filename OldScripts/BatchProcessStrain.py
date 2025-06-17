import os
import re
import numpy as np
import pandas as pd
from OldScripts.ComputingStrain_r4 import main as run_strain_analysis

def extract_image_number(filename):
    match = re.search(r"(\d+)\.avg", filename)
    return int(match.group(1)) if match else None

def main_batch():
    input_dir = "InputFiles"
    output_dir = "StrainOutputs"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for file in os.listdir(input_dir):
        if file.endswith(".avg.tif"):
            full_path = os.path.join(input_dir, file)
            image_number = extract_image_number(file)
            print(f"\n[INFO] Processing: {file} (Image #{image_number})")

            try:
                output_tensor_path = os.path.join(output_dir, f"tensor_{image_number}.npy")
                run_strain_analysis(tif_override=full_path, output_tensor_path=output_tensor_path)

                tensor = np.load(output_tensor_path)
                tensor_flat = tensor.flatten()
                results.append([image_number] + tensor_flat.tolist())
            except Exception as e:
                print(f"[ERROR] Failed to process {file}: {e}")

    # Save results to CSV
    col_names = ["image_number"] + [f"eps{i}{j}" for i in range(1, 4) for j in range(1, 4)]
    df = pd.DataFrame(results, columns=col_names)
    df.sort_values(by="image_number", inplace=True)
    df.to_csv(os.path.join(output_dir, "strain_tensors_summary.csv"), index=False)
    print("\n[INFO] Strain tensor summary saved to 'strain_tensors_summary.csv'")

if __name__ == "__main__":
    main_batch()