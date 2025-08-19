import FunctionLibrary as fl
import time

def map_strain():
    json_path = "OutputFiles_Data_VB-APS-SSAO-6_25C_2025.08.19-15.41.01/strain_tensor_summary.json" # path to the json housing the strain data
    sample_name = "VB-APS-SSAO-6_25C" # mapped sample name
    n_rows = 44 # number of rows
    n_cols = 8 # number of columns
    pixel_size = (0.1, 0.025) # (width, height) in mm

    start_time = time.time()
    batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))

    output_directory = fl.create_directory(
        f"OutputFiles_StrainMaps_{sample_name}_{batch_time_suffix}", 
        logger=None)
    
    map_name_pfx = f"{sample_name}_strain-map"

    fl.generate_strain_maps_from_json(
        json_path=json_path,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=output_directory,
        dpi=600,
        pixel_size=pixel_size,
        map_name_pfx=map_name_pfx,
        logger=None)

if __name__ == "__main__":
    map_strain()