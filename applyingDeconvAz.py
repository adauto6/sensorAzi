import pandas as pd

import matplotlib.pyplot as plt

from scipy.fftpack import ifft, fft 

from scipy import signal

import numpy as np

from sklearn.metrics import r2_score

df = pd.read_csv('analise.csv')

#mu1 = 0.75

mu1 = 0.8

#mu1 = 0.23

sig = 0.05

volumes = [1, 2, 5, 6, 7] # [1, 2, 3, 4, 5, 6, 7]

y_value = [] 

x_value = [0.2, 0.4, 1.0, 1.2, 1.4] # Concentracao Retirada do volume e qtd de azitromicina colocada

for volume in volumes:
	signalVolu = df[f'Av. I2 {volume}0uL'].values #[100:188]

	x = df['E (V)'].values #[100:188]

	gauss1 = (1/(mu1*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu1)/sig)**2)

	gauss1_f = fft(gauss1)

	signal_f = fft(signalVolu)

	signal_f_wo_g1 = signal_f / gauss1_f.conj()

	signal_t_wo_g1 = ifft(signal_f_wo_g1)

	plt.figure(1)

	b, a = signal.butter(4, 0.05, 'low')
	
	zi = signal.lfilter_zi(b, a)
	
	z, _ = signal.lfilter(b, a, signal_t_wo_g1, zi=zi*signal_t_wo_g1[0])
	
	z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
	
	signal_t_wo_g1_filter = signal.filtfilt(b, a, signal_t_wo_g1)

	#plt.plot(x, signal_t_wo_g1, label=f'Deconv{volume}0uL')
	
	y_value.append(max(z2))
	
#	x_value.append(x[z2.index(max(z2))])
	
#	x_value.append(x[np.where(max(z2) == z2)])
	
	plt.plot(x, z2, label=f'Deconv{volume}0uL - filtered')

#	plt.yticks(np.arange(-0.2, 0.4, 0.05)) 
#	plt.xlim(0.1, 1.0)

#	plt.ylim(-0.1, 0.2)

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

	plt.plot(x, signalVolu, label=f'signal {volume}0uL')

	plt.plot(x, signal_conv_t, label='signal conv t')


	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
	plt.tight_layout()
	
	if volume >= 4:
		plt.savefig('signal.png')

print(y_value, x_value)
plt.figure(4)

coef = np.polyfit(x_value,y_value,1)
poly1d_fn = np.poly1d(coef) 
 
#plt.plot(x,y, 'yo', x, poly1d_fn(x_value))
	
y_calc = poly1d_fn(x_value)

realy = []

calcrealy = []

print(y_calc, type(y_calc))

for i,ii in zip(y_value, y_calc):
	realy.append(abs(i))	
	calcrealy.append(abs(ii))	

R2 = r2_score(realy, calcrealy)	
	
plt.plot(x_value, y_value, label='Dados')
plt.plot(x_value, y_calc,  label=f'y={"{:.2f}".format(abs(coef[0]))}*x + {"{:.2f}".format(abs(coef[1]))}, R^2 = {"{:.2f}".format(R2)}')

plt.xlabel('Concetracao (mg/L)')
plt.ylabel('Corrente (apos convolucao e filtro)')
plt.legend()
plt.savefig('fitting.png')


#plt.show()

