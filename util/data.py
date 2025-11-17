import numpy as np

class System:
    def __init__(self, name, images, FITS_file=None, lens=None):
        self.name = name
        self.images = np.array(images)
        self._lens = np.array(lens) if lens is not None else None
        self._FITS_file = FITS_file
    
    @property
    def has_lens(self):
        return self._lens is not None
    
    @property
    def has_FITS(self):
        return self._FITS_file is not None
    
    @property
    def lens(self):
        if self.has_lens:
            return self._lens
        else:
            raise ValueError('Lens data is not available for this system.')
        
    @property
    def FITS_file(self):
        if self.has_FITS:
            return self._FITS_file
        else:
            raise ValueError('FITS file is not available for this system.')
        
    def __repr__(self):
        return f"System(name={self.name}, images={self.images.tolist()}, FITS_file={self._FITS_file}, lens={self._lens.tolist() if self._lens is not None else None})"    
        
class Cluster:
    def __init__(self, name, folder_path, systems):
        self.name = name
        self.folder_path = folder_path
        self.dataset = systems
        
        try: 
            self.systems = datasets.get(systems) # TODO: check systems
        except:
            raise ValueError(f"Dataset '{systems}' does not exist.")
        
        # Check that all systems have the same FITS file
        if self.systems and len(self.systems) > 1:
            fits_files = [system._FITS_file for system in self.systems if system.has_FITS]
            if fits_files and len(set(fits_files)) > 1:
                raise ValueError("All systems in a cluster must have the same FITS file.")
                
            self._FITS_file = fits_files[0] if fits_files else None
                
        self.all_points = np.array([])
        
        for system in self.systems:
            if self.all_points.size == 0:
                self.all_points = np.array(system.images)
            else:
                self.all_points = np.vstack([self.all_points, system.images])
                
        self.center_offset = np.mean(self.all_points, axis=0)
            
    @property
    def has_FITS(self):
        return self._FITS_file is not None
    
    @property
    def FITS_file(self):
        if self.has_FITS:
            return self._FITS_file
        else:
            raise ValueError('FITS file is not available for this cluster.')
            
    def __repr__(self):
        return f"Cluster(name={self.name}, folder_path={self.folder_path}, dataset={self.dataset}, FITS_file={self._FITS_file}, center_offset={[round(float(x), 2) for x in self.center_offset]})"

# --- Datasets ---

clj2325 = [
    System('2', [[529,885], [1050, 455], [481, 192], [217, 669]]),
    System('3', [[375,999], [556, 1052], [1066, 718], [564, 246]]),
    System('6', [[186,990], [835, 1022], [980, 886], [611, 258]])
]

clj2325_FITS = [
    System('2', FITS_file='fits/clj2325/original_data/hst_skycell-p0485x05y14_acs_wfc_f814w_all_drc.fits',
           images=[[3179.1032,4767.2363],[4283.2816,3861.6466],[3097.9145,3263.2509],[2473.3609,4284.0492]]),
    System('3', FITS_file='fits/clj2325/original_data/hst_skycell-p0485x05y14_acs_wfc_f814w_all_drc.fits',
           images=[[2851.446,5034.9925],[3219.6689,5131.4477],[4309.4683,4442.6845],[3259.5525,3398.2619]]),
    System('6', FITS_file='fits/clj2325/original_data/hst_skycell-p0485x05y14_acs_wfc_f814w_all_drc.fits',
           images=[[2415.9157,4975.1417],[3802.5472,5079.4108],[4116.0904,4788.5126],[3356.9939,3426.6775]])
]

a1689_FITS = [
    System('4', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2038.7397,2033.396], [2532.7029,1740.3972], [2463.7202,3015.7972], [3805.2522,2472.7942]]), # e - center img.
    System('8', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2000.6501,2158.9958], [2270.5755,1868.1967], [2242.6801,2898.7967], [4032.8922,2780.5928]]),
    System('9', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2600.1827,3407.7973], [1635.6501,2172.3939], [3069.853,1664.1971], [3807.0529,2642.7943]])
]

a1689_FITS_lim = [
    System('4 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2038.9193,2034.076], [2535.582,1741.7573], [2463.6,3019.0372], [3804.5921,2478.3143]]), # e - center img.
    System('8 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2001.4898,2165.1158], [2273.5747,1869.1967], [2241.1808,2901.6767], [4031.3322,2777.8328]]),
    System('9 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2601.8022,3414.3173], [1640.8694,2173.0339], [3066.0741,1664.7171], [3810.352,2649.6743]])
]

a1689_FITS_centroid = [
    System('4 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2041.4659, 2033.1823], [2535.0029, 1740.4792], [2465.656, 3016.3897], [3807.5621, 2473.7682]]), # e - center img.
    System('8 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2003.3734, 2159.7411], [2251.7953, 1860.3709], [2243.8931, 2898.4897], [4030.3778, 2776.3938]]),
    System('9 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2601.6592, 3408.1493], [1636.1257, 2172.3832], [3069.2807, 1663.4704], [3808.0358, 2643.0179]])
]

a1689_FITS_all = [
    System('1 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[3779.7632,3211.3944], [2759.1395,1832.1974], [1771.5097,2631.3947], [2111.6157,3062.5962]]), # avg a & b, f - center img.
    System('2 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[3733.5763,3270.9947], [1800.6015,2670.5948], [2097.8191,3037.3962], [2747.4429,1859.7974]]), # e - center img.
    System('4 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2038.7397,2033.396], [2532.7029,1740.3972], [2463.7202,3015.7972], [3805.2522,2472.7942]]), # e - center img.
    System('8 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2000.6501,2158.9958], [2270.5755,1868.1967], [2242.6801,2898.7967], [4032.8922,2780.5928]]),
    System('9 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2600.1827,3407.7973], [1635.6501,2172.3939], [3069.853,1664.1971], [3807.0529,2642.7943]]),
    System('12 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[3504.733,2112.2958], [3569.217,2483.5955], [3326.1358,3041.7965], [1584.9643,2202.5935]]), # avg a & b, avg d & f
    System('19 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2200.9921,2727.9965], [4118.3688,2779.7922], [2103.8215,1993.1962], [2079.228,2031.3961]]),
    System('24 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2933.3915,2056.3973], [2073.5252,3369.5961], [2604.081,3696.5974], [1575.6623,2782.5935]]),
    System('29 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[2923.1942,2019.5973], [2684.4599,3696.7974], [2038.1349,3322.996], [1603.5555,2767.9937]]),
    System('42 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
        images=[[3089.3507,3521.1971], [2310.1608,3310.3968], [1638.3483,2462.9939], [2980.4781,1830.9973]])
]

# --- Not used anymore, but kept for reference ---

a1689_all_systems = [
    System('1', [[2213, 1055], [2130, 1040], [1356, 2557], [2499, 3112], [2745, 2621], [1890, 2333]]),
    System('2', [[2245, 1064], [2522, 3069], [2728, 2644], [1386, 2556], [1876, 2349]]),
    System('3', [[2372, 2832], [2279, 2921], [1809, 2984]]),
    System('4', [[1844, 3124], [1369, 2801], [2553, 2322], [1492, 1338], [2052, 2252]]),
    System('5', [[1606, 2205], [1706, 2211], [2751, 3343]]),
    System('6', [[3097, 2066], [2811, 3058], [3055, 2744], [2943, 2709]]),
    System('7', [[1087, 1250], [2439, 2349]]),
    System('8', [[1974, 3105], [1596, 2984], [2541, 2572], [1674, 1001]]),
    System('9', [[2850, 2032], [2141, 3430], [1072, 2347], [1645, 1264]]),
    System('10', [[2189, 3560], [2130, 1625], [2012, 2098]]),
    System('11', [[1824, 3522], [2462, 1801], [2062, 2135]]),
    System('12', [[1275, 1796], [1311, 1730], [1602, 1547], [2197, 1563], [2190, 3463], [2224, 1497]]),
    System('13', [[3610, 2512], [3605, 2567], [3559, 2725]]),
    System('14', [[639, 2648], [677, 2772]]),
    System('15', [[1356, 2557], [2213, 1055], [2009, 2074]]),
    System('16', [[1892, 1715], [1950, 1991], [2326, 3635]]),
    System('17', [[2240, 2437], [2152, 2389], [1209, 1038]]),
    System('18', [[2499, 3112], [2130, 1040], [2029, 2106]]),
    System('19', [[2404, 2682], [1637, 924], [1780, 3082], [1825, 3088]]),
    System('21', [[1907, 2714], [1897, 2650], [1797, 852]]),
    System('22', [[2407, 2038], [2127, 2144], [1539, 3348]]),
    System('23', [[2364, 2005], [2134, 2121], [1582, 3408]]),
    System('24', [[1485, 2304], [3039, 2525], [3110, 1906], [2719, 3225]]),
    System('26', [[1396, 1008], [2317, 2621], [2046, 2389]]),
    System('27', [[1391, 1017], [2334, 2627], [2034, 2378]]),
    System('28', [[2193, 1677], [2062, 3713], [2034, 2080]]),
    System('29', [[1456, 2329], [3076, 1833], [3012, 2577], [2694, 3206]]),
    System('30', [[3642, 2362], [3626, 2623], [3566, 2809]]),
    System('31', [[1890, 2333], [2219, 3312], [2245, 1064], [1760, 1282]]),
    System('32', [[2821, 2671], [2637, 3097], [1416, 2468], [1796, 2361]]),
    System('33', [[1310, 2140], [2745, 2621]]),
    System('35', [[1348, 2159], [2521, 3397], [1906, 2186]]),
    System('36', [[3060, 2352], [3049, 2397]]),
    System('40', [[2421, 2219], [973, 1545]]),
    System('41', [[1453, 1882], [2777, 3795], [1856, 2054]]),
    System('42', [[2745, 1541], [2885, 2336], [2403, 3304], [1261, 2357]]),
    System('44', [[2030, 1825], [2060, 3803]]),
    System('45', [[2522, 3069], [2769, 3852]]),
    System('46', [[1964, 2900], [1711, 796]]),
    System('48', [[2144, 2779], [1660, 868]]),
    System('49', [[2216, 1849], [1852, 3582]]),
    System('50', [[2145, 3117], [1485, 2910], [2570, 2613]])
]

SDSS_J1004_all = [
    System('QSO', [[467.30589556, 481.57390258], [530.50540575, 310.3543409], [-68.50145018, 700.185749], [60.09188893, 12.10719088]]), # QSO-E - center img.
    System('4', [[322.94063644, -507.55155055], [-419.68389052, -297.76612035], [-465.23126053, -229.92693735], [257.55746422, 576.54392506]]),
    System('41', [[349.6843392, -495.80627249], [-409.28604213, -288.18388524], [-466.75504978, -210.04769932], [263.6371458, 597.12121827]]),
    System('42', [[394.3755718, -472.96136888], [-401.00151171, -291.18555757], [-473.64837141, -197.32453294], [266.99877236, 611.29321375]]),
    System('43', [[404.83984549, -456.11930782], [-393.5700173, -304.50216119], [-497.45329499, -156.41502757], [277.30605627, 618.48372464]]) # 43.5 - center img.
]

SDSS_J1004 = [
    System('4', [[322.94063644, -507.55155055], [-419.68389052, -297.76612035], [-465.23126053, -229.92693735], [257.55746422, 576.54392506]]),
    System('41', [[349.6843392, -495.80627249], [-409.28604213, -288.18388524], [-466.75504978, -210.04769932], [263.6371458, 597.12121827]]),
    System('42', [[394.3755718, -472.96136888], [-401.00151171, -291.18555757], [-473.64837141, -197.32453294], [266.99877236, 611.29321375]]),
    System('43', [[404.83984549, -456.11930782], [-393.5700173, -304.50216119], [-497.45329499, -156.41502757], [277.30605627, 618.48372464]]) # 43.5 - center img.
]

original_systems = [
    System('PG1115+080', lens=[[0.381, -1.344]],
            images=[[1.328, -2.034], [1.477,-1.5776], [-0.341, -1.961], [0,0]]),
    System('H1413+117', lens=[[0.142,0.561]],
            images=[[0,0], [0.744, 0.168], [-0.492, 0.713], [0.354, 1.040]]),
    System('B1422+231', lens=[[0.742, -0.656]],
            images=[[0.385, 0.317], [0,0], [-0.336, -0.750], [0.948, -0.802]]),
    System('2M1134-2103', lens=[[-0.74, 0.66]],
            images=[[0.74,1.75], [0,0], [-1.93, -0.77], [-1.23, 1.35]])
]

basic_elements = [
    System('DOZ41', [[120.0, 548.0], [137.5, 530.5], [136.0, 512.0], [73.0, 511.0]]), 
    System('DOZ42', [[229.0, 552.5], [273.0, 530.0], [276.0, 519.0], [222.0, 493.0]]), 
    System('DOZ43', [[338.5, 528.5], [351.0, 493.0], [383.0, 488.0], [405.5, 559.0]]), 
    System('DOZ44', [[461.0, 510.5], [475.0, 493.0], [504.0, 491.0], [509.0, 555.0]]),
    System('DOZ45', [[76.0, 406.5], [85.5, 422.5], [138.0, 418.0], [113.0, 368.0]]), 
    System('DOZ46', [[199.5, 417.0], [228.0, 376.0], [281.0, 367.0], [251.5, 417.5]]),
    System('DOZ47', [[364.0, 367.0], [376.5, 430.0], [394.0, 392.0], [348.0, 400.5]]),
    System('DOZ48', [[522.0, 420.5], [530.0, 419.5], [528.0, 364.5], [458.0, 383.0]]),
    System('DOZ49', [[146.0, 302.0], [108.5, 303.0], [77.0, 268.0], [135.0, 240.0]]),
    System('DOZ50', [[252.0, 260.5], [241.0, 245.0], [229.0, 258.0], [237.0, 303.0]]),
    System('DOZ51', [[345.5, 267.0], [371.0, 288.0], [388.0, 276.0], [373.0, 241.5]]),
    System('DOZ52', [[472.0, 282.5], [467.5, 270.0], [513.0, 240.0], [525.5, 296.0]]) 
]

a1689_7_sisters = [
    System('1', [[2172, 1048], [1356, 2557], [2499, 3112], [2745, 2621]]), # avg a & b, f - center img.
    System('2', [[2245, 1064], [2522, 3069], [2728, 2644], [1386, 2556]]), # e - center img.
    System('4', [[1844, 3124], [1369, 2801], [2553, 2322], [1492, 1338]]), # e - center img.
    System('8', [[1974, 3105], [1596, 2984], [2541, 2572], [1674, 1001]]),
    System('9', [[2850, 2032], [2141, 3430], [1072, 2347], [1645, 1264]]),
    System('12', [[1293, 1763], [1602, 1547], [2211, 1530], [2190, 3463]]), # avg a & b, avg d & f
    System('19', [[2404, 2682], [1637, 924], [1780, 3082], [1825, 3088]])
]

a1689_3_miscreants = [
    System('24', [[1485, 2304], [3039, 2525], [3110, 1906], [2719, 3225]]),
    System('29', [[1456, 2329], [3076, 1833], [3012, 2577], [2694, 3206]]),
    System('42', [[2745, 1541], [2885, 2336], [2403, 3304], [1261, 2357]])
]

a1689_all = [
    System('1', [[2172, 1048], [1356, 2557], [2499, 3112], [2745, 2621]]), # avg a & b, f - center img.
    System('2', [[2245, 1064], [2522, 3069], [2728, 2644], [1386, 2556]]), # e - center img.
    System('4', [[1844, 3124], [1369, 2801], [2553, 2322], [1492, 1338]]), # e - center img.
    System('8', [[1974, 3105], [1596, 2984], [2541, 2572], [1674, 1001]]),
    System('9', [[2850, 2032], [2141, 3430], [1072, 2347], [1645, 1264]]),
    System('12', [[1293, 1763], [1602, 1547], [2211, 1530], [2190, 3463]]), # avg a & b, avg d & f
    System('19', [[2404, 2682], [1637, 924], [1780, 3082], [1825, 3088]]),
    System('24', [[1485, 2304], [3039, 2525], [3110, 1906], [2719, 3225]]),
    System('29', [[1456, 2329], [3076, 1833], [3012, 2577], [2694, 3206]]),
    System('42', [[2745, 1541], [2885, 2336], [2403, 3304], [1261, 2357]])
    # System('6', [[3097, 2066], [2811, 3058], [3055, 2744], [2943, 2709]]), # doesn't have 2 on each branch
    # System('31', [[1890, 2333], [2219, 3312], [2245, 1064], [1760, 1282]]), # doesn't have 2 on each branch
    # System('32', [[2821, 2671], [2637, 3097], [1416, 2468], [1796, 2361]]), # weird behavior (basically two split images)
]

a1689_FITS_all_lim = [
    System('1 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[3778.7919,3218.8856], [2759.9077,1833.9685], [1771.6184,2634.6056], [2111.3637,3067.3272]]), # e - center img.
    System('2 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[3733.0846,3279.0059], [1803.2895,2674.9258], [2099.8469,3042.8472], [2748.3909,1862.0485]]),
    System('4 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2601.8022,3414.3173], [1640.8694,2173.0339], [3066.0741,1664.7171], [3810.352,2649.6743]]),
    System('8 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2201.5921,2729.5965], [4114.8296,2785.7522], [2105.1412,1997.3562], [2077.0691,2036.9561]]),
    System('9 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2038.9193,2034.076], [2535.582,1741.7573], [2463.6,3019.0372], [3804.5921,2478.3143]]), # e - center img.
    System('19 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2001.4898,2165.1158], [2273.5747,1869.1967], [2241.1808,2901.6767], [4031.3322,2777.8328]]),
    System('24 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2601.8022,3414.3173], [1640.8694,2173.0339], [3066.0741,1664.7171], [3810.352,2649.6743]]),
    System('29 FITS', FITS_file='fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits',
           images=[[2201.5921,2729.5965], [4114.8296,2785.7522], [2105.1412,1997.3562], [2077.0691,2036.9561]])
]

# --- Datasets and Clusters ---

datasets = {
    'clj2325': clj2325,
    'clj2325_fits': clj2325_FITS,
    'a1689_fits': a1689_FITS,
    'a1689_fits_lim': a1689_FITS_lim,
    'a1689_fits_centroid': a1689_FITS_centroid,
    'a1689_fits_all': a1689_FITS_all,
    'j1004': SDSS_J1004,
    'a1689_7_sisters': a1689_7_sisters,
    'a1689_3_miscreants': a1689_3_miscreants,
    'a1689_all': a1689_all,
    'a1689_fits_all_lim': a1689_FITS_all_lim,
    'basic_elements': basic_elements
}

clusters = {
    'a1689': Cluster('Abell 1689', 'a1689', 'a1689_fits'),
    'a1689_lim': Cluster('Abell 1689', 'a1689', 'a1689_fits_lim'),
    'a1689_centroid': Cluster('Abell 1689', 'a1689', 'a1689_fits_centroid'),
    'a1689_all': Cluster('Abell 1689', 'a1689', 'a1689_fits_all'),
    'a1689_all_lim': Cluster('Abell 1689', 'a1689', 'a1689_fits_all_lim'),
    'clj2325_pixels': Cluster('SPT-CLJ2325-4111', 'clj2325', 'clj2325'),
    'clj2325': Cluster('SPT-CLJ2325-4111', 'clj2325', 'clj2325_fits'),
}