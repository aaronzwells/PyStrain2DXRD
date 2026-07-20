import FunctionLibrary as fl
import numpy as np
import logging

def main():
    fityk_input_pfx = "/Users/benjaminschneiderman/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Code/Python/Analysis_pyFAI-1/2_BinnedIntegrationAndFitting/VB-APS-SSAO-6_25C_TestMap-AO_000492/BinnedOutput/Fityk/mid_azim_"
    chi = np.linspace(1.5, 358.5, 120)
    bkg_spline = [15.98, 36.81, 19.66, 20.96, 23.68, 15.15, 28.26, 11.62, 43.09, 9.33, 48.74, 9.16, 54.32, 5.46, 57.1, 2.47, 61.2, 0.88]
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
    fl.write_fityk_script("test_fityk_script.fit", fityk_input_pfx, chi, initial_q_guesses, tol_up, tol_down, bkg_spline=bkg_spline)

if __name__ == "__main__":
    main()