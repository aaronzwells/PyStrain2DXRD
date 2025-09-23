import FunctionLibrary as fl
import time

def map_strain():
    json_path = "OutputFiles_Data_VB-APS-SSAO-6_800C_2025.09.22-09.05.11/strain_tensor_summary.json"
    sample_name = "VB-APS-SSAO-6_800C"
    n_rows = 21
    n_cols = 4
    num_gap_cols = 1
    pixel_size = (0.1, 0.050)

    start_time = time.time()
    batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))

    output_directory = fl.create_directory(
        f"OutputFiles_StrainMaps_{sample_name}_{batch_time_suffix}", 
        logger=None)
    
    map_name_pfx = f"{sample_name}_strain-map"

    fl.generate_strain_maps_from_json_nonContinuous(
        json_path=json_path,
        n_rows=n_rows,
        n_cols=n_cols,
        gap_cols=num_gap_cols,
        gap_mm=None,
        output_dir=output_directory,
        dpi=600,
        pixel_size=pixel_size,
        map_name_pfx=map_name_pfx,
        logger=None)

if __name__ == "__main__":
    map_strain()