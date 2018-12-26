import sys
import numpy as np
import matplotlib.pyplot as plt

'''
任意のパワースペクトルと振幅特性をもつ IR 測定用スイープ生成


応答と逆特性の直線状畳み込みにも対応。
逆特性は自分で作ること。

------
使い方 
------
N: タップ数
J: 実効長
magnitude_dB: パワースペクトル（デシベル）
envelope_dB: 振幅特性（デシベル）

* 振幅特性: スイープ信号の時間波形において，その周波数の瞬間の振幅


--------
参考文献
--------

中原優樹, 金田豊,
``インパルス応答測定結果の帯域別雑音レベルを一定とする効率的な残響時間測定法,''
日本音響学会誌, 72(7), pp. 358--366, 2016.

金田豊,
``音響インパルス応答測定用信号について,''
信学技法, EA2015-68, pp. 13--18, 2015.

守谷直也, 金田豊,
``雑音に起因する誤差を最小化するインパルス応答測定信号,''
日本音響学会誌, 64(12), pp. 695--701, 2008.

Swen Muller,
``Transfer-function measurement with sweeps,''
J Audio Eng Soc, 49(6), pp. 443--471, 2001.

鈴木陽一 et al.,
``時間引き延ばしパルスの設計法に関する考察,''
信学技法, EA92-86, pp. 17--24, 1992.

'''



def gen_sweep(N, J, magnitude_dB, envelope_dB, plot=False, downward=False):
    '''
    if (int(N) & int(N - 1)) != 0: 
        print('N should be a pewer of 2.', file=sys.stderr)
    if type(N) != int:
        N = int(N)
    if magnitude_dB.shape != (N // 2 + 1, ):
        print('magnitude should be shape (N/2+1,).', file=sys.stderr)
    if envelope_dB.shape != (N // 2 + 1, ):
        print('envelope should be shape (N/2+1,).', file=sys.stderr)
    '''
    # power
    E = 10 ** (magnitude_dB / 10) / 10 ** (envelope_dB / 10)

    # group delay
    gd = np.cumsum(E)
    gd *= J / gd[-1]
    
    # phase
    ph = np.cumsum(gd) * 2. * np.pi / N
    ph *= np.around(ph[N // 2] / np.pi) * np.pi / ph[N // 2]
    shift = np.linspace(0, np.pi, N // 2 + 1, endpoint=True) * int((N - J) / 2)
    ph += shift
    if downward == True:
        ph *= -1

    # amplitude
    A = np.sqrt(E * (J * N) / 2 / (2 * np.sum(E) - E[0] - E[N // 2])) # amp. 1
    A *= 10 ** (envelope_dB / 20) # desired envelope
    
    # IFFT
    SS = A * np.exp(-1.j * ph)
    ss = np.fft.irfft(SS)
    
    # plot
    if plot == True:
        fs = 48000
        f = np.linspace(0, fs / 2, N // 2 + 1, endpoint=True)
        plt.figure(figsize=(5, 10))
        
        plt.subplot(411)
        plt.plot(f, magnitude_dB, lw=1)
        plt.title('mag. sweep'); plt.xlabel('Hz'); plt.ylabel('dB')
        plt.xlim([10, 24000]); plt.xscale('log'); plt.grid(which='both')
        
        plt.subplot(412)
        plt.plot(f, envelope_dB, lw=1)
        plt.title('desired envelope'); plt.xlabel('Hz'); plt.ylabel('dB')
        plt.xlim([10, 24000]); plt.xscale('log'); plt.grid(which='both')
        
        plt.subplot(413)
        plt.plot(f, gd / fs, lw=1)
        plt.title('delay sweep'); plt.xlabel('Hz'); plt.ylabel('sec')
        plt.xlim([10, 24000]); plt.xscale('log'); plt.grid(which='both')
        
        plt.subplot(414)
        plt.plot(np.arange(ss.shape[0]) / fs, ss, lw=0.5)
        plt.title('sweep'); plt.xlabel('sec'); plt.ylabel('V')
        plt.grid(which='both')
        
        plt.tight_layout()
        #plt.show()
        
    return ss



'''
from scipy.signal import fftconvolve
from .io import writewav

def example():

    fs = 48000

    N = 2 ** 18
    J = N / 2
    
    f = np.linspace(0, fs / 2, N // 2 + 1, endpoint=True)
    
    # almost 1 / f
    magnitude = 10. * np.log10(1. / (1. + (f / 1))) # [dB]
    
    # -6dB/oct from over 3000 Hz
    envelope = 10. * np.log10(1. / (1. + (f / 3000) ** 2)) - 3 # [dB]
    
    
    ss = gen_sweep(N, J, magnitude, envelope, plot=True, downward=False)
    
    plt.show()

    return



if __name__ == '__main__':
    
    example()
'''
