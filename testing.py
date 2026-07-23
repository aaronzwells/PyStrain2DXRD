import FunctionLibrary as fl
import numpy as np
import logging
import os

def demo_fityk_script_one_bin():
    scanid = 304 #Zero strain position for room temp map
    sample_name = "VB-APS-SSAO-6_25C_Map-AO"
    scan_name = f"{sample_name}_{scanid:06d}"
    fityk_input_pfx = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", scan_name, "BinnedOutput/Fityk/mid_azim_")
    chi = 1.5
    # For ID 304:
    bkg_spline = [15.12,4.59, 16.22,2.49, 18.02,0.56, 20.45,0.03, 29.31,-0.06, 43.33,-0.03, 56.6,-0.06, 61.17,-0.06, 65.05,-0.03]
    initial_q_guesses = [ # AL2O3 EXPERIMENT VALID INITIAL GUESSES (Correct peak pos.)
            18.103087,
            24.677268,
            26.458500,
            30.203330,
            36.188437,
            39.321810,
            44.830282,
            45.838482
        ]
    tol_up = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    tol_down = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    script_name = f"demo_fityk_script_scanid_{scanid}_bin_{chi:.1f}deg.fit"
    fl.write_fityk_script(script_name, fityk_input_pfx, [chi], initial_q_guesses, tol_up, tol_down, bkg_spline=bkg_spline)

def demo_fityk_script_all_bins():
    scanid = 304 #Zero strain position for room temp map
    sample_name = "VB-APS-SSAO-6_25C_Map-AO"
    scan_name = f"{sample_name}_{scanid:06d}"
    fityk_input_pfx = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", scan_name, "BinnedOutput/Fityk/mid_azim_")
    chi = np.linspace(1.5, 358.5, 120)
    # For ID 304:
    bkg_spline = [15.12,4.59, 16.22,2.49, 18.02,0.56, 20.45,0.03, 29.31,-0.06, 43.33,-0.03, 56.6,-0.06, 61.17,-0.06, 65.05,-0.03]
    initial_q_guesses = [ # AL2O3 EXPERIMENT VALID INITIAL GUESSES (Correct peak pos.)
            18.103087,
            24.677268,
            26.458500,
            30.203330,
            36.188437,
            39.321810,
            44.830282,
            45.838482
        ]
    tol_up = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    tol_down = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    script_name = f"demo_fityk_script_scanid_{scanid}_bins_all.fit"
    fl.write_fityk_script(script_name, fityk_input_pfx, chi, initial_q_guesses, tol_up, tol_down, bkg_spline=bkg_spline)


def main():
    demo_fityk_script_one_bin()
    # demo_fityk_script_all_bins()

    # sample_name = "VB-APS-SSAO-6_25C_Map-AO"
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
    
    # #Store the q0 centroids
    # sample_name = "VB-APS-SSAO-6_25C_Map-AO"
    # q0_reference_scanid = 304
    # q0_reference_scan_name = f"{sample_name}_{q0_reference_scanid:06d}"
    # q0_reference_dir = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", q0_reference_scan_name)
    # q0_reference_fityk_input_pfx = os.path.join(os.getcwd(), "2_BinnedIntegrationAndFitting", q0_reference_scan_name, "BinnedOutput/Fityk/mid_azim_")
    # q0_centroids_arr = fl.format_fityk_outputs(q0_reference_fityk_input_pfx, chi, num_peaks=len(hkl), output_dir=q0_reference_dir)  #Warning, num_peaks hard coded for now

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

if __name__ == "__main__":
    main()