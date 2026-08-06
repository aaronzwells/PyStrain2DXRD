import yaml
import h5py
import numpy as np
    
with h5py.File("Ceria_1137.h5", "r") as f:
    print("Demo OmegaSumFrame shape:", f["OmegaSumFrame"].shape)
    print("Demo Q_map shape:", f["geometry_maps/Q_map"].shape)
    print("Demo Eta_map shape:", f["geometry_maps/Eta_map"].shape)

with h5py.File("your_output.h5", "r") as f:
    print("Yours OmegaSumFrame shape:", f["OmegaSumFrame"].shape)
    print("Yours Q_map shape:", f["geometry_maps/Q_map"].shape)
    print("Yours Eta_map shape:", f["geometry_maps/Eta_map"].shape)

with h5py.File("Ceria_1137.h5", "r") as f:
    q_map = f["geometry_maps/Q_map"][:]
    eta_map = f["geometry_maps/Eta_map"][:]

# If Q varies along axis 0 (rows) and is constant along axis 1 (columns),
# that confirms Q_map[:, 0] sweeps through all q values, matching Q_map.shape[0] = npt_rad
print("Q varies along axis 0?", not np.allclose(q_map[:, 0], q_map[:, 1]))
print("Q constant along axis 1?", np.allclose(q_map[0, :], q_map[0, :].mean()))

print("Eta varies along axis 1?", not np.allclose(eta_map[0, :], eta_map[1, :]))
print("Eta constant along axis 0?", np.allclose(eta_map[:, 0], eta_map[:, 0].mean()))


# chi = np.linspace(1.5, 358.5, 120)

    # #Placeholders - look up actual hkl later
    # hkl = [(1, 0, 0), 
    #        (2, 0, 0), 
    #        (3, 0, 0), 
    #        (4, 0, 0),
    #        (5, 0, 0),
    #        (6, 0, 0),
    #        (7, 0, 0),
    #        (8, 0, 0)
    #        ]
    
    # # #Store the q0 centroids
    # sample_name = "VB-APS-SSAO-6_25C_Map-AO"
    # q0_reference_scanid = 304
    # q0_reference_scan_name = f"{sample_name}_{q0_reference_scanid:06d}"
    # q0_reference_dir = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", q0_reference_scan_name)
    # # q0_reference_fityk_input_pfx = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", q0_reference_scan_name, "BinnedOutput/Fityk/mid_azim_")
    # q0_centroids_arr = fl.format_fityk_outputs(q0_reference_scanid, sample_name, chi, num_peaks=len(hkl), output_dir=q0_reference_dir)  #Warning, num_peaks hard coded for now
    # print(q0_centroids_arr)

    # sample_name = "VB-APS-SSAO-6_25C_TestMap-AO"
    # scanid = 492
    # scan_name = f"{sample_name}_{scanid:06d}"
    # scan_output_dir = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", scan_name)
    # scan_reference_fityk_input_pfx = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", scan_name, "BinnedOutput/Fityk/mid_azim_")
    # q_centroids_arr = fl.format_fityk_outputs(scan_reference_fityk_input_pfx, chi, num_peaks=len(hkl), output_dir=scan_output_dir)  #Warning, num_peaks hard coded for now
    
    # #Keep Aaron's stacked format by hkl
    # fl.plot_q_vs_chi_stacked(q_centroids_arr, source_of_fit="fityk", output_dir=scan_output_dir, chi_deg=chi)
    # fl.visualize_distortion_combined_hkl(q0_centroids_arr, q_centroids_arr, q0_reference_scanid, scanid, hkl, chi_deg=chi, output_dir=scan_output_dir, dpi=600, calibrant=False)


    # #Reset the parameters for the non-zero strain location
    # sample_name = "VB-APS-SSAO-6_25C_TestMap-AO"
    # scanid = 492
    # scan_name = f"{sample_name}_{scanid:06d}"
    # output_dir = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", scan_name)
    # chi = np.linspace(1.5, 358.5, 120)

# def main():
#     demo_fityk_script_one_bin()

#     demo_fityk_script_all_bins()

# if __name__ == "__main__":
#     main()