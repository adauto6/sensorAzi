import pandas as pd

import matplotlib.pyplot as plt

from scipy.fftpack import ifft, fft 

from scipy import signal

import numpy as np

df = pd.read_csv('analise.csv')

#mu1 = 0.75

mu1 = 0.23

sig = 0.05

volumes = [1, 2, 3, 4, 5, 6, 7]

for volume in volumes:
	signalVolu = df[f'Av. I2 {volume}0uL'].values[100:188]

	x = df['E (V)'].values[100:188]

	gauss1 = (1/(mu1*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu1)/sig)**2)

	gauss1_f = fft(gauss1)

	signal_f = fft(signalVolu)

	signal_f_wo_g1 = signal_f / gauss1_f.conj()

	signal_t_wo_g1 = ifft(signal_f_wo_g1)

	plt.figure(1)

	b, a = signal.butter(4, 0.13, 'low')
	
	zi = signal.lfilter_zi(b, a)
	
	z, _ = signal.lfilter(b, a, signal_t_wo_g1, zi=zi*signal_t_wo_g1[0])
	
	z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
	
	signal_t_wo_g1_filter = signal.filtfilt(b, a, signal_t_wo_g1)

#	plt.plot(x, signal_t_wo_g1, label=f'Deconv{volume}0uL')
	
	plt.plot(x, z2, label=f'Deconv{volume}0uL - filtered')

	#plt.yticks(np.arange(-0.2, 0.4, 0.05)) 
	#plt.xlim(0.1, 1.0)

	#plt.ylim(-0.1, 0.2)

	plt.legend()
	if volume >= 4:
		plt.savefig('deconv.png', dpi=1200)

	plt.figure(2)

	plt.plot(x, gauss1, label='gauss1')

	plt.legend()
	if volume >= 4:
		plt.savefig('gauss1.png')

	plt.figure(3)

	signal_conv_f = signal_f_wo_g1 * gauss1_f.conj()

	signal_conv_t = ifft(signal_conv_f)

	plt.plot(x, signalVolu, label='signal')

	plt.plot(x, signal_conv_t, label='signal conv t')

	plt.legend()
	if volume >= 4:
		plt.savefig('signal.png')

#plt.show()

