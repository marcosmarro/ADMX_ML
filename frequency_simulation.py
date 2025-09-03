import numpy as np
import pyfftw


def transfer_func(f,f_0, Q):
    factor1     = f_0**2 / f**2
    transferred = 1 / ((1 - factor1)**2 * Q**2 + factor1)
    return transferred
    

def f_v_divided_by_v(v):
    v_solar  = 232e3 # m/s
    v_0      = 220e3 # m/s 
    v_escape = 550e3 # 1e3 
    f_v = 1 / (np.sqrt(np.pi) * v_solar * v_0) * np.exp(-((v + v_solar) / v_0) ** 2) * (np.exp(4*v*v_solar / v_0**2) - 1) 
    
    if isinstance(f_v, float):
        if v > v_escape or v <0: f_v = 0
        return f_v 
    else:
        # f_v[v > v_escape] = 0
        f_v[v <= 0] = 0
        return f_v 


def averaged_power(x):
    avg_power = np.mean(x*x)
    return avg_power 


def f_v(v):
    return f_v_divided_by_v(v) * v 


def lorentz(x,x0,Q):
    return 1/(4*(x-x0)*(x-x0)*Q*Q/(x0*x0)+1)


def reflection(f,f0,Q,beta):
    Q_0 = Q*(1.+beta)
    delta=Q_0*(f-f0)/f0
    return (beta-1.-1.j*(2.*delta))/(beta+1.+1.j*(2.*delta))


def Gamma2(f,f0,Q,beta):
    reflec_coe = reflection(f,f0,Q,beta)
    return np.abs(np.conj(reflec_coe)*reflec_coe)


def psd_to_time(psd):
    N_pos = len(psd)
    phase = np.random.rand(N_pos)*2*np.pi 
    print('phase:', phase)
 
    voltage = np.sqrt(psd)*np.exp(1j*phase)
    pos_freqs = voltage
    if N_pos % 2 != 0:
        #even
        X_full = np.concatenate([pos_freqs, pos_freqs[-2:0:-1].conj()])
    else:
        #odd N_pos
        X_full = np.concatenate([pos_freqs, pos_freqs[-1:0:-1].conj()])

    time_series = pyfftw.interfaces.scipy_fftpack.ifft(X_full)
    
    return np.append(time_series, [0])


def make_injected(input_len, sig_size, detune, scale_factor):
    """Creates axion time series signal derived from https://arxiv.org/abs/2408.04696.
    """

    N_per_sec = int(400e3) # sampling rate ()
    dt    = 1. / N_per_sec #sampling delta time sec
    f_max = 1 / dt # nominal maximum frequency after fft
    vector_freq = np.linspace(0, f_max/2, int(input_len / 2), endpoint=False) # only the first half is meaningful
    f_digi = 100e3 # Hz equivalent to down-conversion

    h_plank  = 6.62607015e-34 # J*sec
    E_a0_ueV = 4 # ueV
    c_v      = 3e8 # m/s
    E_a0     = E_a0_ueV * 1e-6 * 1.6e-19 # 
    m_a      = E_a0 / c_v**2

    beta = 2.

    f_0_signal = E_a0/h_plank 
    print(f"signal info: f_0_signal = {f_0_signal/1e9:.3e} GHz; E_a0 = {E_a0_ueV} ueV")
    f_cav = f_0_signal + detune
    vector_freq_up_convert = vector_freq + f_cav - f_digi
    E_kinetic = h_plank * vector_freq_up_convert - E_a0 
    E_kinetic[E_kinetic < 0]=0
    velocities = np.sqrt(2 * E_kinetic / m_a)
    velocities = np.array(velocities.real, dtype=np.float64) / scale_factor
    vector_psd_signal_part = f_v_divided_by_v(velocities) * scale_factor ** 2
    normal_factor = sum(f_v(velocities))
    vector_psd_signal_part= vector_psd_signal_part / normal_factor

    QL = 50000
    thermal_bkg = 0
    
    ##every bin should follow an exponential distribution
    ##If there is no signal, the bkg mean is normalized to 1.
    ##If there is a signal in the bin the mean is related the signal distribution function
    vector_sig=sig_size*4e3*QL*beta/(1+beta)*lorentz(vector_freq_up_convert,f_cav, QL) * vector_psd_signal_part #signal 
    
    total_mean_exponential = thermal_bkg + vector_sig
    psd_samples_0 = np.random.exponential(scale=1, size=len(vector_freq))

    psd_samples = psd_samples_0 * total_mean_exponential
    voltage_time_series_simu = psd_to_time(psd_samples) * np.sqrt(N_per_sec)

    return voltage_time_series_simu.real