#For documentation purpose, `inline` is used to enforce the storage of the image in the notebook
#matplotlib widget
#many imports which will be used all along the notebook
import time
import pyFAI
import fabio
import numpy
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from math import sin, cos, sqrt
from scipy.ndimage import convolve, binary_dilation
from scipy.optimize import minimize
#from matplotlib.pyplot import subplots
from pyFAI.ext.bilinear import Bilinear
from pyFAI.ext.watershed import InverseWatershed
from silx.resources import ExternalResources
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.spatial.distance import cdist


def main():
    start_time = time.perf_counter()
    print("Using pyFAI version: ", pyFAI.version)

    # A couple of compound dtypes ...
    dt = numpy.dtype([('y', numpy.float64),
                    ('x', numpy.float64),
                    ('i', numpy.int64),
                    ])
    dl = numpy.dtype([('y', numpy.float64),
                    ('x', numpy.float64),
                    ('i', numpy.int64),
                    ('Y', numpy.int64),
                    ('X', numpy.int64),
                    ])

    ring_file = "Pilatus2MCdTe_ID15_CeO2_72100eV_800mm_0000.cbf"
    print(ring_file)
    grid_file = "Pilatus2MCdTe_ID15_grid_plus_sample_0004.cbf"
    print(grid_file)


    rings = fabio.open(ring_file).data
    img = fabio.open(grid_file).data
    fig,ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img.clip(0,1000), interpolation="bilinear")
    ax[0].set_title("grid")
    ax[1].imshow(numpy.arcsinh(rings), interpolation="bilinear")
    ax[1].set_title("rings")
    plt.show()
    plt.close()
    #pass

    # This is the default detector as definied in pyFAI according to the specification provided by Dectris:
    pilatus = pyFAI.detector_factory("Pilatus_2m_CdTe")
    print(pilatus)

    mask1 = pilatus.mask
    module_size = pilatus.MODULE_SIZE
    module_gap = pilatus.MODULE_GAP
    submodule_size = (96,60)

    #1 + 2 Calculation of the module_id and the interpolated-mask:
    mid = numpy.zeros(pilatus.shape, dtype=int)
    mask2 = numpy.zeros(pilatus.shape, dtype=int)
    idx = 1
    for i in range(8):
        y_start = i*(module_gap[0] + module_size[0])
        y_stop = y_start + module_size[0]
        for j in range(3):
            x_start = j*(module_gap[1] + module_size[1])
            x_stop = x_start + module_size[1]
            mid[y_start:y_stop,x_start: x_start+module_size[1]//2] = idx
            idx+=1
            mid[y_start:y_stop,x_start+module_size[1]//2: x_stop] = idx
            idx+=1
            mask2[y_start+submodule_size[0]-1:y_start+submodule_size[0]+2,
                x_start:x_stop] = 1
            for k in range(1,8):
                mask2[y_start:y_stop,
                x_start+k*(submodule_size[1]+1)-1:x_start+k*(submodule_size[1]+1)+2] = 1

    #Extra masking
    mask0 = img<0
    #Those pixel are miss-behaving... they are the hot pixels next to the beam-stop
    mask0[915:922,793:800] = 1
    mask0[817:820,747:750] = 1

    fig,ax = plt.subplots(1,3, figsize=(10,4))
    ax[0].imshow(mid, interpolation="bilinear")
    ax[0].set_title("Module Id")

    ax[1].imshow(mask2+mask1+mask0, interpolation="bilinear")
    ax[1].set_title("Combined mask")

    nimg = img.astype(float)
    nimg[numpy.where(mask0+mask1+mask2)] = numpy.nan


    ax[2].imshow(nimg)#, interpolation="bilinear")
    ax[2].set_title("Nan masked image")
    plt.show()
    plt.close()
    #pass

    def sliding_window_view(x, shape, subok=False, readonly=True):
        """
        Creates sliding window views of the N dimensional array with the given window
        shape. Window slides across each dimension of `x` and extract subsets of `x`
        at any window position.
        Parameters
        ----------
        x : array_like
            Array to create sliding window views of.
        shape : sequence of int
            The shape of the window. Must have same length as the number of input array dimensions.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the returned
            array will be forced to be a base-class array (default).
        readonly : bool, optional
            If set to True, the returned array will always be readonly view.
            Otherwise it will return writable copies(see Notes).
        Returns
        -------
        view : ndarray
            Sliding window views (or copies) of `x`. view.shape = x.shape - shape + 1
        See also
        --------
        as_strided: Create a view into the array with the given shape and strides.
        broadcast_to: broadcast an array to a given shape.
        Notes
        -----
        ``sliding_window_view`` create sliding window views of the N dimensions array
        with the given window shape and its implementation based on ``as_strided``.
        Please note that if readonly set to True, views are returned, not copies
        of array. In this case, write operations could be unpredictable, so the returned
        views are readonly. Bear in mind that returned copies (readonly=False) will
        take more memory than the original array, due to overlapping windows.
        For some cases there may be more efficient approaches to calculate transformations
        across multi-dimensional arrays, for instance `scipy.signal.fftconvolve`, where combining
        the iterating step with the calculation itself while storing partial results can result
        in significant speedups.
        Examples
        --------
        >>> i, j = np.ogrid[:3,:4]
        >>> x = 10*i + j
        >>> shape = (2,2)
        >>> np.lib.stride_tricks.sliding_window_view(x, shape)
        array([[[[ 0,  1],
                [10, 11]],
                [[ 1,  2],
                [11, 12]],
                [[ 2,  3],
                [12, 13]]],
            [[[10, 11],
                [20, 21]],
                [[11, 12],
                [21, 22]],
                [[12, 13],
                [22, 23]]]])
        """
        np = numpy
        # first convert input to array, possibly keeping subclass
        x = np.array(x, copy=False, subok=subok)

        try:
            shape = np.array(shape, dtype=np.int64)
        except Exception:
            raise TypeError('`shape` must be a sequence of integer')
        else:
            if shape.ndim > 1:
                raise ValueError('`shape` must be one-dimensional sequence of integer')
            if len(x.shape) != len(shape):
                raise ValueError("`shape` length doesn't match with input array dimensions")
            if np.any(shape <= 0):
                raise ValueError('`shape` cannot contain non-positive value')

        o = np.array(x.shape) - shape  + 1 # output shape
        if np.any(o <= 0):
            raise ValueError('window shape cannot larger than input array shape')

        if not isinstance(readonly, bool):
            raise TypeError('readonly must be a boolean')

        strides = x.strides
        view_strides = strides

        view_shape = np.concatenate((o, shape), axis=0)
        view_strides = np.concatenate((view_strides, strides), axis=0)
        view = as_strided(x, view_shape, view_strides, subok=subok, writeable=not readonly)

        if not readonly:
            return view.copy()
        else:
            return view

    start_time_bkg = time.time()
    #Calculate a background image using a large median filter ... takes a while
    shape = (19,11)
    print(nimg.shape)
    padded = numpy.pad(nimg, tuple((i//2,) for i in shape), mode="edge")
    print(padded.shape)
    background = numpy.nanmedian(sliding_window_view(padded, shape), axis = (-2,-1))
    print(background.shape)
    fig,ax = plt.subplots()
    ax.imshow(background)
    ax.set_title("Background image")
    print("Background image calculated in %.2f seconds" % (time.time()-start_time_bkg))

    plt.show()
    plt.close()
    #pass

    fig,ax = plt.subplots(1,2, figsize=(9,5))

    normalized = (nimg/background)

    low = numpy.nanmin(normalized)
    high = numpy.nanmax(normalized)
    print(low, high)
    normalized[numpy.isnan(normalized)] = 0
    normalized /= high

    ax[0].imshow(normalized)
    ax[0].set_title("Normalized image")

    ax[1].hist(normalized.ravel(), 100, range=(0,1))
    ax[1].set_title("Histogram of intensities in normalized image")

    plt.show()
    plt.close()
    #pass

    #print the profile of the normalized image: the center is difficult to measure due to the small size of the hole.
    fig,ax = plt.subplots(2)
    ax[0].plot(normalized[:,545])
    ax[1].plot(normalized[536,:])
    plt.show()
    plt.close()
    #pass

    #Definition of the convolution kernel
    ksize = 5
    y,x = numpy.ogrid[-(ksize-1)//2:ksize//2+1,-(ksize-1)//2:ksize//2+1]
    d = numpy.sqrt(y*y+x*x)

    #Fade out curve definition
    def fadeout(x):
        return 1/(1+numpy.exp(5*(x-2.5)))

    kernel = fadeout(d)
    mini=kernel.sum()
    print(mini)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(d)
    ax[0].set_title("Distance array")

    ax[1].plot(numpy.linspace(0,5,100),fadeout(numpy.linspace(0,5,100)))
    ax[1].set_title("fade-out curve")

    ax[2].imshow(kernel)
    ax[2].set_title("Convolution kernel")
    plt.show()
    plt.close()
    #pass

    my_smooth = convolve(normalized, kernel, mode="constant", cval=0)/mini
    print(my_smooth.shape)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(normalized.clip(0,1))
    ax[0].set_ylim(1050,1100)
    ax[0].set_xlim(300,350)
    ax[1].imshow(my_smooth.clip(0,1))
    ax[1].set_ylim(1050,1100)
    ax[1].set_xlim(300,350)
    numpy.where(my_smooth == my_smooth.max())
    plt.show()
    plt.close()
    #pass

    all_masks = numpy.logical_or(numpy.logical_or(mask0,mask1),mask2)
    print(all_masks.sum())
    big_mask = binary_dilation(all_masks, iterations=ksize//2+1+1)
    print(big_mask.sum())
    smooth2 = my_smooth.copy()
    smooth2[big_mask] = 0
    fig,ax = plt.subplots()
    ax.imshow(smooth2)
    plt.show()
    plt.close()
    #pass

    #Display the profile of the smoothed image: the center is easy to measure thanks to the smoothness of the signal
    fig,ax = plt.subplots(2)
    ax[0].plot(my_smooth[:,545])
    ax[1].plot(my_smooth[536,:])
    plt.show()
    plt.close()
    #pass

    iw = InverseWatershed(my_smooth)
    iw.init()
    iw.merge_singleton()
    all_regions = set(iw.regions.values())

    regions = [i for i in all_regions if i.size>mini]

    print("Number of region segmented: {}".format(len(all_regions)))
    print("Number of large enough regions : {}".format(len(regions)))

    #Remove peaks on masked region
    sieved_region = [i for i in regions if not big_mask[(i.index//nimg.shape[-1], i.index%nimg.shape[-1])]]
    print("Number of peaks not on masked areea : {}".format(len(sieved_region)))

    # Histogram of peak height:
    s = numpy.array([i.maxi for i in sieved_region])

    fig, ax = plt.subplots()
    ax.hist(s, 100)
    plt.show()
    plt.close()
    #pass

    #sieve-out for peak intensity
    int_mini = 0.1
    peaks = [(i.index//nimg.shape[-1], i.index%nimg.shape[-1]) for i in sieved_region if (i.maxi)>int_mini]
    print("Number of remaining peaks with I>{}: {}".format(int_mini, len(peaks)))

    peaks_raw = numpy.array(peaks)

    # Finally the peak positions are interpolated using a second order taylor expansion 
    # in thevinicy of the maximum value of the signal:

    #Create bilinear interpolator
    bl = Bilinear(my_smooth)

    #Overlay raw peak coordinate and refined peak positions

    ref_peaks = [bl.local_maxi(p) for p in peaks]
    fig, ax = plt.subplots()
    ax.imshow(img.clip(0,1000), interpolation="nearest")
    peaks_ref = numpy.array(ref_peaks)
    ax.plot(peaks_raw[:,1], peaks_raw[:, 0], ".r")
    ax.plot(peaks_ref[:,1],peaks_ref[:, 0], ".b")
    ax.set_title("Extracted peak position (red: raw, blue: refined)")
    print("Refined peak coordinate:")
    print(ref_peaks[:10])

    yxi = numpy.array([i+(mid[round(i[0]),round(i[1])],) 
                   for i in ref_peaks], dtype=dt)
    print("Number of keypoint per module:")
    for i in range(1,mid.max()+1):
        print("Module id:",i, "cp:", (yxi[:]["i"] == i).sum())

    # pairwise distance calculation using scipy.spatial.distance.cdist

    dist = cdist(peaks_ref, peaks_ref)

    fig, ax = plt.subplots()
    ax.hist(dist.ravel(), 100, range=(0,100))
    ax.set_title("Pair-wise distribution function")

    #from pair-wise distribution histogram
    step = 29 
    #work with the first module and fit the peak positions
    first = yxi[yxi[:]["i"] == 1]
    y_min = first[:]["y"].min()
    x_min = first[:]["x"].min()
    print("offset for the first peak: ", x_min, y_min)

    #Assign each peak to an index
    indexed1 = numpy.zeros(len(first), dtype=dl)

    for i,v in enumerate(first):
        Y = round((v["y"]-y_min)/step)
        X = round((v["x"]-x_min)/step)
        indexed1[i]["y"] = v["y"]
        indexed1[i]["x"] = v["x"]
        indexed1[i]["i"] = v["i"]
        indexed1[i]["Y"] = Y
        indexed1[i]["X"] = X
        print(f'peak id: {i} {v:20s} Y:{Y} (Δ={(v["y"]-Y*step-y_min)/step:.3f}) X:{X} (Δ={(v["x"]-X*step-x_min)/step:.3f})')

    #Calculate the center of every single module for rotation around this center.
    centers = {i: numpy.array([[numpy.where(mid == i)[1].mean()], [numpy.where(mid == i)[0].mean()]]) for i in range(1, 49)}
    for k,v in centers.items():
        print(k,v.ravel())

    # Define a rotation of a module around the center of the module ...

    def rotate(angle, xy, module):
        "Perform the rotation of the xy points around the center of the given module"
        rot = [[cos(angle),-sin(angle)],
            [sin(angle), cos(angle)]]
        center = centers[module]
        return numpy.dot(rot, xy - center) + center
    
    guess1 = [step, y_min, x_min, 0]

    def cost1(param):
        """contains: step, y_min, x_min, angle for the first module
        returns the sum of distance squared in pixel space
        """
        step = param[0]
        y_min = param[1]
        x_min = param[2]
        angle = param[3]
        XY = numpy.vstack((indexed1["X"], indexed1["Y"]))
    #     rot = [[cos(angle),-sin(angle)],
    #            [sin(angle), cos(angle)]]
        xy_min = [[x_min], [y_min]]
        xy_guess = rotate(angle, step * XY + xy_min, module=1)
        delta = xy_guess - numpy.vstack((indexed1["x"], indexed1["y"]))
        return (delta*delta).sum()
    
    print("Before optimization", guess1, "cost=", cost1(guess1))
    res1 = minimize(cost1, guess1, method = "slsqp")
    print(res1)
    print("After optimization", res1.x, "cost=", cost1(res1.x))
    print("Average displacement (pixels): ",sqrt(cost1(res1.x)/len(indexed1)))

    #retrieve the result of the first module fit:
    step, y_min, x_min, angle = res1.x
    indexed = numpy.zeros(yxi.shape, dtype=dl)

    # rot =  [[cos(angle),-sin(angle)],
    #         [sin(angle), cos(angle)]]
    # irot =  [[cos(angle), sin(angle)],
    #          [-sin(angle), cos(angle)]]

    print("cost1: ",cost1([step, y_min, x_min, angle]), "for:", step, y_min, x_min, angle)

    xy_min = numpy.array([[x_min], [y_min]])
    xy = numpy.vstack((yxi["x"], yxi["y"]))
    indexed["y"] = yxi["y"]
    indexed["x"] = yxi["x"]
    indexed["i"] = yxi["i"]
    XY_app = (rotate(-angle, xy, 1)-xy_min) / step
    XY_int = numpy.round((XY_app)).astype("int")
    indexed["X"] = XY_int[0]
    indexed["Y"] = XY_int[1]
    xy_guess = rotate(angle, step * XY_int + xy_min, 1)

    thres = 1.2
    delta = abs(xy_guess - xy)
    print((delta>thres).sum(), "suspicious peaks:")
    suspicious = indexed[numpy.where(abs(delta>thres))[1]]
    print(suspicious)

    fig,ax = plt.subplots()
    ax.imshow(img.clip(0,1000))
    ax.plot(indexed["x"], indexed["y"],".g")
    ax.plot(suspicious["x"], suspicious["y"],".r")
    plt.show()
    plt.close()
    #pass

    def submodule_cost(param, module=1):
        """contains: step, y_min_1, x_min_1, angle_1, y_min_2, x_min_2, angle_2, ...
        returns the sum of distance squared in pixel space
        """
        
        step = param[0]
        y_min1 = param[1]
        x_min1 = param[2]
        angle1 = param[3]
        
        mask = indexed["i"] == module
        substack = indexed[mask]
        
        XY = numpy.vstack((substack["X"], substack["Y"]))
    #     rot1 = [[cos(angle1), -sin(angle1)],
    #             [sin(angle1), cos(angle1)]]
        xy_min1 = numpy.array([[x_min1], [y_min1]])
        xy_guess1 = rotate(angle1, step * XY + xy_min1, module=1)
        #This is guessed spot position for module #1
        if module == 1:
            "Not much to do for module 1"
            delta = xy_guess1 - numpy.vstack((substack["x"], substack["y"]))
        else:
            "perform the correction for given module"
            y_min = param[(module-1)*3+1]
            x_min = param[(module-1)*3+2]
            angle = param[(module-1)*3+3]     

    #         rot = numpy.array([[cos(angle),-sin(angle)],
    #                            [sin(angle), cos(angle)]])
            xy_min = numpy.array([[x_min], [y_min]])
            xy_guess = rotate(angle, xy_guess1+xy_min, module)
            delta = xy_guess - numpy.vstack((substack["x"], substack["y"]))

        return (delta*delta).sum()

    guess145 = numpy.zeros(48*3+1)
    guess145[:4] = res1.x
    for i in range(1, 49):
        print("Cost for module #",i, submodule_cost(guess145, i))

    def total_cost(param):
        """contains: step, y_min_1, x_min_1, angle_1, ...
        returns the sum of distance squared in pixel space
        """
        return sum(submodule_cost(param, module=i) for i in range(1,49))
    total_cost(guess145)

    print("Before optimization", guess145[:10], "cost=", total_cost(guess145))
    res_all = minimize(total_cost, guess145, method = "slsqp")
    print(res_all)
    print("After optimization", res_all.x[:10], "cost=", total_cost(res_all.x))

    for i in range(1,49):
        print(f"Module id: {i} cost: {submodule_cost(res_all.x, i):.3f} Δx: {res_all.x[-2+i*3]:.3f}, Δy: {res_all.x[-1+i*3]:.3f} rot: {numpy.rad2deg(res_all.x[i*3]):.3f}°")

    def correct(x, y, dx, dy, angle, module):
        "apply the correction dx, dy and angle to those pixels ..."
        trans = numpy.array([[dx],
                            [dy]])
        xy_guess = numpy.vstack((x.ravel(), 
                                y.ravel()))
        xy_cor = rotate(-angle, xy_guess, module) - trans
        xy_cor = xy_cor.reshape((2,)+x.shape)
        return xy_cor[0], xy_cor[1]
    

    pixel_coord = pyFAI.detector_factory("Pilatus2MCdTe").get_pixel_corners()
    pixel_coord_raw = pixel_coord.copy()
    for i in range(2, 49):
        # Extract the pixel corners for one module
        module_idx = numpy.where(mid == i)
        one_module = pixel_coord_raw[module_idx]
        #retrieve the fitted values
        dy, dx, angle = res_all.x[-2+i*3:1+3*i]
        
        y = one_module[..., 1]/pilatus.pixel1
        x = one_module[..., 2]/pilatus.pixel2
        
        #apply the correction the other way around
        x_cor, y_cor = correct(x, y, dx, dy, angle, i)
        one_module[...,1] = y_cor * pilatus.pixel1 #y
        one_module[...,2] = x_cor * pilatus.pixel2 #x
        #Update the array
        pixel_coord_raw[module_idx] = one_module

    pilatus.set_pixel_corners(pixel_coord_raw)
    pilatus.mask = all_masks
    pilatus.save("Pilatus_ID15_raw.h5")
    
    displ = numpy.sqrt(((pixel_coord - pixel_coord_raw)**2).sum(axis=-1))
    displ /= pilatus.pixel1 #convert in pixel units
    fig, ax = plt.subplots()
    ax.hist(displ.ravel(), 100)
    ax.set_title("Displacement of pixels versus the reference representation")
    ax.set_xlabel("Error in pixel size (172µm)")
    plt.show()
    plt.close()
   #pass

    misaligned = numpy.vstack((pixel_coord_raw[..., 2].ravel(), #x
                            pixel_coord_raw[..., 1].ravel())) #y

    reference = numpy.vstack((pixel_coord[..., 2].ravel(), #x
                            pixel_coord[..., 1].ravel())) #y
    
    #Kabsch alignment of the whole detector ... 

    def kabsch(P, R):
        "Align P on R"
        centroid_P = P.mean(axis=0)
        centroid_R = R.mean(axis=0)
        centered_P = P - centroid_P
        centered_R = R - centroid_R
        C = numpy.dot(centered_P.T, centered_R)
        V, S, W = numpy.linalg.svd(C)
        d = (numpy.linalg.det(V) * numpy.linalg.det(W)) < 0.0
        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        # Create Rotation matrix U
        U = numpy.dot(V, W)
        P = numpy.dot(centered_P, U)
        return P + centroid_R
        
    aligned = kabsch(misaligned.T, reference.T).T

    displ = numpy.sqrt(((aligned-reference)**2).sum(axis=0))
    displ /= pilatus.pixel1 #convert in pixel units
    fig, ax = plt.subplots()
    ax.hist(displ.ravel(), 100)
    ax.set_title("Displacement of pixels versus the reference representation")
    ax.set_xlabel("Pixel size (172µm)")
    plt.show()
    plt.close()
    #pass

    pixel_coord_aligned = pixel_coord.copy()
    pixel_coord_aligned[...,1] = aligned[1,:].reshape(pixel_coord.shape[:-1])
    pixel_coord_aligned[...,2] = aligned[0,:].reshape(pixel_coord.shape[:-1])

    pilatus.set_pixel_corners(pixel_coord_aligned)
    pilatus.mask = all_masks
    pilatus.save("Pilatus_ID15_Kabsch.h5")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow((pixel_coord_aligned[...,2].mean(axis=-1) - pixel_coord[...,2].mean(axis=-1))/pilatus.pixel2)
    ax[0].set_title("Displacement x (in pixel)")
    ax[1].imshow((pixel_coord_aligned[...,1].mean(axis=-1) - pixel_coord[...,1].mean(axis=-1))/pilatus.pixel1)
    ax[1].set_title("Displacement y (in pixel)")
    plt.show()
    plt.close()
    #pass

    # The geometry has been obtained from pyFAI
    geo = { "dist":  0.8001094657585498,
            "poni1": 0.14397714477803805,
            "poni2": 0.12758748978422835,
            "rot1":  0.0011165686147339689,
            "rot2":  0.0002214091645638961,
            "rot3":  0,
            "detector": "Pilatus2MCdTe"}
    ai_unc = AzimuthalIntegrator(**geo)
    geo["detector"] = "Pilatus_ID15_Kabsch.h5"
    ai_cor = AzimuthalIntegrator(**geo)
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    method = ("pseudo", "histogram", "cython")
    res_unc = ai_unc.integrate2d_ng(rings, 100, 100, radial_range=(7.9, 8.2), unit="2th_deg", method=method, mask=all_masks)
    res_cor = ai_cor.integrate2d_ng(rings, 100, 100, radial_range=(7.9, 8.2), unit="2th_deg", method=method, mask=all_masks)
    opts = {"origin":"lower",
            "extent": [res_unc.radial.min(), res_unc.radial.max(), -180, 180], 
            "aspect":"auto",
            "cmap":"inferno"}
    ax[0].imshow(res_unc[0], **opts)
    ax[1].imshow(res_cor[0], **opts)
    ax[0].set_xlabel(r"Scattering angle 2$\theta$ ($^{o}$)")
    ax[1].set_xlabel(r"Scattering angle 2$\theta$ ($^{o}$)")
    ax[0].set_ylabel(r"Azimuthal angle $\chi$ ($^{o}$)")
    ax[0].set_title("Uncorrected")
    ax[1].set_title("Corrected")
    plt.show()
    plt.close()
    #pass

    print(f"Total execution time: {time.perf_counter()-start_time:.3f}s")

if __name__ == "__main__":
    main()