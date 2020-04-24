# coding: utf-8
import numpy as np
import audioproc as ap


def calc_decaycurve(ir, inf=-1):
    decay = 10 * np.log10(np.flip(np.cumsum(np.flip(ir[:inf]) ** 2)))
    decay -= decay[0]
    return decay


def arg_attenuate(decaycurve, att, start=0, relative=False):
    '''
    returns the smallest index of the element below "-att" in the decay-curve,
    and returns -1 if there are no elements under "-att".
    
    * The decay-curve is monotonically decreasing.
    * search range is right of "start".
    * If "relative" is set to True, search for an index that
      attenuates att based on the attenuation of start.
    '''
    
    if relative:
        target = decaycurve[start] - att
    else:
        target = -att

    if np.max(decaycurve) <= target:
        return start
    elif np.min(decaycurve) > target:
        return -1

    L = len(decaycurve) - 1
    i = start + (L - start) // 2
    
    step = (L - start) // 4
    while step > 0:
        if decaycurve[i] > target:
            i += step
        else:
            i -= step
        step = step // 2
    
    while decaycurve[i] < target:
        i -= 1
    
    while decaycurve[i] > target:
        i += 1
        
    return i


def calc_RT(decaycurve, att1=5, att2=35, fs=48000):
    '''
    reverberation time
    '''

    # line fitting
    i1 = arg_attenuate(decaycurve, att1)
    i2 = arg_attenuate(decaycurve, att2, start=i1)
    x = np.arange(i1, i2)
    tilt, intercept = np.polyfit(x, decaycurve[i1:i2], 1)
    
    # RT
    RT = (-60 / tilt) / fs
    
    # standard deviation
    fitcurve = tilt * x + intercept
    SD = np.sqrt(np.mean((fitcurve - decaycurve[i1:i2]) ** 2))

    return RT, SD


def calc_clarity(ir, msec=80, fs=48000):
    p_sq = ir ** 2
    i0 = np.argmax(p_sq)
    i1 = i0 + int(fs * msec / 1000)
    return 10 * np.log10(np.sum(p_sq[i0:i1]) / np.sum(p_sq[i1:]))


def calc_deutlichkeit(ir, msec=50, fs=48000):
    p_sq = ir ** 2
    i0 = np.argmax(p_sq)
    i1 = i0 + int(fs * msec / 1000)
    return np.sum(p_sq[i0:i1]) / np.sum(p_sq)
    
