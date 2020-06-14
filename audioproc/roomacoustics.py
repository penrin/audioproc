# coding: utf-8
import numpy as np
import audioproc as ap


def calc_decaycurve(ir, integ_lim=-1):
    decay = 10 * np.log10(np.flip(np.cumsum(np.flip(ir[:integ_lim]) ** 2)))
    decay -= decay[0]
    return decay


def arg_decay(decaycurve, att, start=0, relative=False):
    '''
    returns i that gives decaycurve[i-1] < -att <= decaycurve[i],
    and returns -1 if there are no elements under "-att".

    * The decay-curve must be monotonically decreasing.
    * search range is right of "start".
    * If "relative" is set to True, search -att relative to decaycurve[start].
    '''

    if relative:
        target = decaycurve[start] - att
    else:
        target = -att

    if decaycurve[-1] > target:
        return -1

    i = decaycurve.size - np.searchsorted(
            decaycurve[start:][::-1], target, 'right')
    return i


def calc_RT(decaycurve, att1=5, att2=35, fs=48000):
    '''
    reverberation time
    '''

    # line fitting
    i1 = arg_decay(decaycurve, att1)
    i2 = arg_decay(decaycurve, att2, start=i1)
    x = np.arange(i1, i2)
    tilt, intercept = np.polyfit(x, decaycurve[i1:i2], 1)
    
    # RT
    RT = (-60 / tilt) / fs
    
    # standard deviation
    fitcurve = tilt * x + intercept
    SD = np.sqrt(np.mean((fitcurve - decaycurve[i1:i2]) ** 2))

    return RT, SD


def arg_arrival(ir, trigger=-20):
    '''
    detects direct sound arrival time.
    The threshold is "trigger" smaller than the peak of IR.
    '''
    if trigger > 0:
        raise Exception('trigger > 0')
    p_abs = np.abs(ir)
    i_peak = np.argmax(p_abs)
    threshold = p_abs[i_peak] * 10 ** (trigger / 20)
    i = np.where(p_abs[:i_peak + 1] >= threshold)[0][0]
    return i


