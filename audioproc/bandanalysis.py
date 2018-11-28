import numpy as np
import math

'''
https://en.wikipedia.org/wiki/Octave_band
https://www.onosokki.co.jp/HP-WK/c_support/newreport/noise/souon_11.htm
http://kikakurui.com/c1/C1514-2002-01.html
JIS C 1514
'''


'''
center and limit frequencies of (1 / octN) octave band
'''
def bandfreq_base2(octN, fc_lim=(16, 20000), f_ref=1000):
    octave_ratio = 2
    if (octN % 2) == 1: # odd
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * np.log2(fc_lim[0] / f_ref) + 1 / 2
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * np.log2(fc_lim[1] / f_ref) - 1 / 2
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** (n / octN)
    else: # even
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * np.log2(fc_lim[0] / f_ref)
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * np.log2(fc_lim[1] / f_ref) - 1
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** ((2 * n + 1) / (2 * octN))
    # limit frequency
    f_lower = f_center * octave_ratio ** (-1 / (2 * octN))
    f_upper = f_center * octave_ratio ** (1 / (2 * octN))
    return f_center, f_lower, f_upper


def bandfreq_base10(octN, fc_lim=(16, 20000), f_ref=1000):
    octave_ratio = 10 ** 0.3 # 1.9953
    if (octN % 2) == 1: # odd
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * math.log(fc_lim[0] / f_ref, octave_ratio) + 1 / 2
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * math.log(fc_lim[1] / f_ref, octave_ratio) - 1 / 2
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** (n / octN)
    else: # even
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * math.log(fc_lim[0] / f_ref, octave_ratio)
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * math.log(fc_lim[1] / f_ref, octave_ratio) - 1
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** ((2 * n + 1) / (2 * octN))
    # limit frequency
    f_lower = f_center * octave_ratio ** (-1 / (2 * octN))
    f_upper = f_center * octave_ratio ** (1 / (2 * octN))
    return f_center, f_lower, f_upper 



