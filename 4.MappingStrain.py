# 4b.MappingStrain-nonContinuous.py

import FunctionLibrary as fl
import time

def map_strain():
    # --- User-Defined Inputs ---
    json_path = "OutputData/OutputFiles_Data_VB-APS-SSAO-6_400C_2025.10.10-15.00.43/strain_tensor_summary.json"
    sample_name = "VB-APS-SSAO-6_400C_test"
    solved_strain_components = 5 # 3 = biaxial; 5 = biaxial w/ shear; 6 = all components
    
    # Define the geometric and measurement parameters for mapping
    n_steps_x = 4      # Number of measurement points in the X direction (columns)
    n_steps_y = 21     # Number of measurement points in the Y direction (rows)
    dX = 0.2           # Center-to-center distance in X (mm)
    dY = 0.05         # Center-to-center distance in Y (mm)
    pixel_size_map = (0.1, 0.05) # Define the size of each colored pixel in the heatmap (width, height) in mm
    start_xy = (0.0, 1.0) # Physical starting coordinate (center of the top-left pixel); (startX, startY) in mm
    gap_mm = None # If an additional gap is added between scanned columns, define it here. Usually this is "None"
    map_offset_xy = (-0.05, -start_xy[1]+pixel_size_map[1]/2) # vector for shifting the map data
    trim_edges = False # allows the user to trim the pixels left and down from the translated (0,0)
    color_limit_window = (0.2, 0.8) # Sets the x-range (in mm) used to determine the color scale limits
    colorbar_scale = (-3.5e-04, 4e-04) # Sets the scale of strain; if uniform scale not desired: None

    # --- Script Execution ---
    start_time = time.time()
    batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))

    if colorbar_scale:
        output_directory = fl.create_directory(
            f"OutputMaps_unifiedColorScale/OutputFiles_StrainMaps_{sample_name}_{batch_time_suffix}")
    else:
        output_directory = fl.create_directory(
            f"OutputMaps_defaultColorScale/OutputFiles_StrainMaps_{sample_name}_{batch_time_suffix}")
    
    map_name_pfx = f"{sample_name}_strain-map"

    fl.generate_strain_maps_from_json(
        json_path=json_path,
        n_rows=n_steps_y,
        n_cols=n_steps_x,
        step_size=(dX, dY),
        pixel_size_map=pixel_size_map,
        start_xy=start_xy,
        gap_mm=gap_mm,
        map_offset_xy=map_offset_xy,
        trim_edges=trim_edges,
        color_limit_window=color_limit_window,
        colorbar_scale=colorbar_scale,
        output_dir=output_directory,
        dpi=600,
        map_name_pfx=map_name_pfx,
        num_strain_components=solved_strain_components)

if __name__ == "__main__":
    map_strain()