import pyFAI, fabio
print("\n\npyFAI version:", pyFAI.version)
img = fabio.open("calibration/ceria_lab6_exsitu_71p676keV_1145mm_100x100_3s_002265avg.tif")
print("Image:", img)

ai = pyFAI.load("calibration/Calibration_LaB6_100x100_3s_r0.poni")
print("\nIntegrator: \n", ai)

img_array = img.data
print("img_array:", type(img_array), img_array.shape, img_array.dtype)
mask = img_array>4e9

res = ai.integrate1d_ng(img_array,
                        1000,
                        mask=mask,
                        unit="q_nm^-1",
                        filename="integrated.dat")