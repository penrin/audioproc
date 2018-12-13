import sys
import numpy as np
import audioproc as ap
from scipy.special import spherical_jn, spherical_yn, sph_harm


def acn_index(N):
    '''
    ACN ordering
    n: order, m: degree
    '''
    L = (int(np.floor(N)) + 1) ** 2
    n_list = np.empty(L, dtype=np.int16)
    m_list = np.empty(L, dtype=np.int16)
    i = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            #print(n, m)
            n_list[i] = n
            m_list[i] = m
            i += 1
    return n_list, m_list



def hv_index(H, V):
    '''
    return n & m of #H#V Mixed-order Ambisonics
    
    Chris Travis, "A New Mixed-order Scheme for Ambisonics signals",
    Ambisonics Symposium 2009
    '''
    n_tmp, m_tmp = acn_index(H)
    v = n_tmp - np.abs(m_tmp)
    i = np.where(v <= V)[0]
    n_list = np.copy(n_tmp[i])
    m_list = np.copy(m_tmp[i])
    return n_list, m_list



def sph_harm_realvalued(m, n, theta, phi): 
    if m < 0:
        Y = np.sqrt(2) * (-1) * np.imag(sph_harm(m, n, theta, phi))
    elif m == 0:
        Y = np.real(sph_harm(m, n, theta, phi))
    elif m > 0:
        Y = np.sqrt(2) * (-1) ** int(m) * np.real(sph_harm(m, n, theta, phi))    
    return Y


def spherical_hn1(n, z):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def spherical_hn2(n, z):
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)




class EncodeMatrix:
    
    def setup_micarray(self, x, y, z, alpha=1):
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        self.theta = np.arctan2(y, x)
        self.phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        self.alpha = alpha
        return

    def encodematrix(self, order, wavenum):

        print('Calc. encode matrix')

        n, m = acn_index(order)
        
        # reshape
        r_ = self.r.reshape(-1, 1, 1)
        theta_ = self.theta.reshape(-1, 1, 1)
        phi_ = self.phi.reshape(-1, 1, 1)
        n_ = n.reshape(1, -1, 1)
        m_ = m.reshape(1, -1, 1)
        k_ = np.array(wavenum).reshape(1, 1, -1)

        # spherical bessel function matrix
        if self.alpha == 1:
            J = spherical_jn(n_, k_ * r_)
        else:
            J = self.alpha * spherical_jn(n_, k_ * r_)\
                    - 1.j * (1 - self.alpha)\
                    * spherical_jn(n_, k_ * r_, derivative=True)

        # Spherical function matrix
        Y = np.empty([r_.shape[0], n_.shape[1]], dtype=np.float)
        for i in range(len(m)):
            Y[:, i] = sph_harm_realvalued(m[i], n[i], self.theta, self.phi)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        
        # Encoding matrix
        JY = J * Y 
        Enc = np.empty([JY.shape[1], JY.shape[0], JY.shape[2]], dtype=JY.dtype)
        for i in range(JY.shape[2]):
            ap.progressbar(i, JY.shape[2])
            Enc[:, :, i] = np.linalg.pinv(JY[:, :, i])
        ap.progressbar(1)
        
        return Enc




class DecodeMatrix:

    def setup_loudspeakerarray(self, x, y, z):
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        self.theta = np.arctan2(y, x)
        self.phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        return

    def decodematrix(self, order, wavenum, nearfieldmodel=False):
        if nearfieldmodel:
            Dec = self._decodematrix_nearfield(order, wavenum)
        else:
            Dec = self._decodematrix_planewave(order, wavenum)
        return Dec
    
    def _decodematrix_planewave(self, order, wavenum):

        print('Calc. decode matrix (plane wave model)')
        n, m = acn_index(order)
        
        print('Not yet support nearfieldmodel=False')
        
        return


    def _decodematrix_nearfield(self, order, wavenum):

        print('Calc. decode matrix (near field model)')
        n, m = acn_index(order)

        # reshape
        r_ = self.r.reshape(1, -1, 1)
        theta_ = self.theta.reshape(1, -1, 1)
        phi_ = self.phi.reshape(1, -1, 1)
        n_ = n.reshape(-1, 1, 1)
        m_ = m.reshape(-1, 1, 1)
        k_ = np.array(wavenum).reshape(1, 1, -1)

        # Decoding matrix
        Y = np.empty([n_.shape[0], r_.shape[1]], dtype=np.float)
        for i in range(len(m)):
            Y[i, :] = sph_harm_realvalued(m[i], n[i], self.theta, self.phi)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        H = spherical_hn2(n_, k_ * r_)
        C = 1j * k_ * H * Y

        Dec = np.empty([C.shape[1], C.shape[0], C.shape[2]], dtype=C.dtype)
        for i in range(C.shape[2]):
            ap.progressbar(i, C.shape[2])
            Dec[:, :, i] = np.linalg.pinv(C[:, :, i])
        ap.progressbar(1)

        return Dec


