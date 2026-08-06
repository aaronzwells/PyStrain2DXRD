import json
import numpy as np
import os

def analyze_strain_data(json_file_path, output_file_path, rows, cols, excluded_cols, excluded_cells):
    """
    Reads strain data, maps it to a 2D grid, and calculates statistics.
    
    Arguments:
    - excluded_cols: A list of column indices (0-based) to ignore completely.
    - excluded_cells: A list of specific (row, col) tuples to ignore.
    """
    
    # 1. Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Validate Grid Size
    total_points = len(data)
    if total_points != rows * cols:
        print(f"WARNING: JSON contains {total_points} entries, but grid is defined as {rows}x{cols} ({rows*cols} spots).")

    components = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']
    global_averages = {comp: [] for comp in components}

    print(f"Processing grid: {rows} rows x {cols} cols")
    print(f"Excluding columns: {excluded_cols}")

    skipped_count = 0
    
    # 3. Iterate through data
    for i, entry in enumerate(data):
        # Calculate row and column from linear index (0-based)
        current_row = i // cols
        current_col = i % cols
        
        # --- EXCLUSION LOGIC ---
        # Check if the entire column is excluded
        if current_col in excluded_cols:
            skipped_count += 1
            continue

        # Check if this specific cell is excluded
        if (current_row, current_col) in excluded_cells:
            skipped_count += 1
            continue
            
        # Proceed with processing
        strain_list = entry.get('strain_tensor', [])
        if not strain_list:
            continue

        for comp in components:
            values = [ring.get(comp, np.nan) for ring in strain_list]
            values_arr = np.array(values, dtype=float)
            
            with np.errstate(invalid='ignore'):
                avg_val = np.nanmean(values_arr)
            
            global_averages[comp].append(avg_val)

    # 4. Determine Max and Min
    results = {}
    for comp in components:
        all_avgs = global_averages[comp]
        valid_avgs = [x for x in all_avgs if not np.isnan(x)]
        
        if valid_avgs:
            results[comp] = {
                'min': np.min(valid_avgs),
                'max': np.max(valid_avgs),
                'count': len(valid_avgs)
            }
        else:
            results[comp] = {'min': None, 'max': None, 'count': 0}

    # 5. Write results
    try:
        with open(output_file_path, 'w') as f:
            f.write("Strain Tensor Summary: Column Filtered Analysis\n")
            f.write("===============================================\n\n")
            f.write(f"Source File: {json_file_path}\n")
            f.write(f"Grid Dimensions: {rows} x {cols}\n")
            f.write(f"Excluded Columns: {excluded_cols}\n")
            f.write(f"Total Valid Locations Processed: {results['eps_xx']['count']}\n\n")
            
            for comp in components:
                res = results[comp]
                f.write(f"Component: {comp}\n")
                if res['count'] > 0:
                    f.write(f"  Min Average: {res['min']:.6e}\n")
                    f.write(f"  Max Average: {res['max']:.6e}\n")
                else:
                    f.write("  No valid data found.\n")
                f.write("-" * 30 + "\n")
                
        print(f"Success! Processed {total_points} entries.")
        print(f"Skipped {skipped_count} excluded locations.")
        print(f"Report written to: {output_file_path}")
        
    except IOError as e:
        print(f"Error writing to file: {e}")

# --- Configuration ---
if __name__ == "__main__":
    input_json = 'OutputData/OutputFiles_Data_VB-APS-SSAO-6_25C_2025.10.29-15.09.53/strain_tensor_summary.json'
    output_txt = 'OutputData/OutputFiles_Data_VB-APS-SSAO-6_25C_2025.10.29-15.09.53/strain_min_max_report.txt'
    
    # 1. Matrix Size
    NUM_ROWS = 44   
    NUM_COLS = 8   
    
    # 2. Exclude Entire Columns (0-based index)
    # e.g., [0, 9] would exclude the first and last columns
    EXCLUDED_COLS = [0,1] 
    
    # 3. Exclude Specific Cells (optional additional filtering)
    # Format: (row, col)
    EXCLUDED_CELLS = [
    ]
    
    analyze_strain_data(input_json, output_txt, NUM_ROWS, NUM_COLS, EXCLUDED_COLS, EXCLUDED_CELLS)