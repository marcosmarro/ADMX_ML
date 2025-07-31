import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import scipy.fft


def f_v_divided_by_v(v):
    v_solar = 232*1e3#m/s
    v_0 = 220*1e3#m/s
    v_escape = 550e3#1e3
    f_v = 1./(np.sqrt(np.pi)*v_solar*v_0)*np.exp(-((v+v_solar)/v_0)**2)*(np.exp(4*v*v_solar/v_0**2)-1)
    if isinstance(f_v, float):
        if v > v_escape or v <0: f_v = 0
        return f_v
    else:
        f_v[v>v_escape] = 0
        f_v[v<=0]=0
        return f_v


def f_v(v):
    return f_v_divided_by_v(v)*v


def lorentz(x,x0,Q):
    return 1/(4*(x-x0)*(x-x0)*Q*Q/(x0*x0)+1)


def reflection(f,f0,Q,beta):
    Q_0 = Q*(1.+beta)
    delta=Q_0*(f-f0)/f0
    return (beta-1.-1.j*(2.*delta))/(beta+1.+1.j*(2.*delta))


def Gamma2(f,f0,Q,beta):
    reflec_coe = reflection(f,f0,Q,beta)
    return np.conj(reflec_coe)*reflec_coe


def psd_to_time(psd):
    N_pos = len(psd)
    if N_pos % 2 != 0:
        psd = psd[:-1]
        N_pos = int(N_pos - 1)
    phase = np.random.rand(N_pos)*2*np.pi 
    voltage = np.sqrt(psd)*np.exp(1j*phase)
    N1 = 2 * (N_pos - 1)
    X_full = np.append(voltage,np.conj(voltage[1:-1][::-1]))
    time_series = pyfftw.interfaces.scipy_fftpack.ifft(X_full)/len(X_full)
    return time_series.real


f_digi = 100e3 # z equivalent to down-conversion
h_plank =  6.62607015e-34 # J*sec
c_v = 3e8  # m/s
QL=30000#
beta = 2.


# def make_injected(length, sig_size, E_a0_ueV, f_detune):

#     N_per_sec = int(400e3) # sampling rate ()
#     dt = 1./N_per_sec # sampling delta time sec
#     f_max = 1/dt # nominal maximum frequency after fft
#     vector_freq = np.linspace(0,f_max/2,int(length/2),endpoint=False) # only the first half is meaningful
    
#     E_a0 = E_a0_ueV*1e-6*1.6e-19 # J
#     m_a = E_a0/c_v**2
#     f_0_signal = E_a0/h_plank 

#     f_cav = f_0_signal+f_detune
#     vector_freq_up_convert = vector_freq+f_cav-f_digi

#     E_kinetic = h_plank * vector_freq_up_convert - E_a0 
#     velocities = np.sqrt(2*E_kinetic/m_a)
#     velocities[E_kinetic<0]=0
#     velocities = np.array(velocities.real,dtype=np.float64)
#     vector_psd_signal_part = f_v_divided_by_v(velocities)

#     normal_factor = sum(f_v(velocities))
#     vector_psd_signal_part= vector_psd_signal_part/normal_factor

#     Temp_off = 0.5 #ratio to the on-res temperature
#     Temp_on = 2.0
#     power_reflec_ratio = Gamma2(vector_freq_up_convert,f_cav, QL,beta)
#     on_res_bkg = Temp_on*(1.-power_reflec_ratio)
#     off_res_bkg = Temp_off * power_reflec_ratio

#     thermal_bkg = on_res_bkg + off_res_bkg 
#     thermal_bkg = 0

#     ## every bin should follow an exponential distribution
#     ## If there is no signal, the bkg mean is normalized to 1.
#     ## If there is a signal in the bin the mean is related the signal distribution function
  
#     vector_sig=sig_size*4e3*QL*beta/(1+beta)*lorentz(vector_freq_up_convert,f_cav, QL) * vector_psd_signal_part #signal 

#     total_mean_exponential = thermal_bkg + vector_sig

#     psd_samples_0 = np.random.exponential(scale=1, size=len(vector_freq))
#     psd_samples = psd_samples_0 * total_mean_exponential


#     voltage_time_series_simu = psd_to_time(psd_samples)

#     return voltage_time_series_simu

def make_injected(length, frequency, amplitude):

    fs = 400e3
    t = np.arange(0, length) / fs
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return signal

# wtf= make_injected(4e5, 1e5, .2)
# fft = pyfftw.interfaces.scipy_fftpack.rfft(wtf)
# plt.plot(fft)
# plt.show()