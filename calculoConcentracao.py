import pandas as pd

import matplotlib.pyplot as plt

from scipy.fftpack import ifft, fft 

from scipy import signal

import numpy as np

from sklearn.metrics import r2_score

# filenames = AguaRIo.csv  analiseAguaLagoa.csv  analise.csv  poco.csv  torneira.csv
# analise.csv --> para obter a equacao da reta 

filenames = ['AguaRIo', 'analiseAguaLagoa',  'analise', 'poco',  'torneira']

coef_a = 1.11e-5
coef_b = 1.82e-5 

conc_az = 786.02  # g/mol

for filename in filenames:
	df = pd.read_csv(filename+'.csv')
	mu1 = 0.8

	sig = 0.05
	
	signalVolu = df[f'Av. I2 0uL'].values[100:188]

	x = df['E (V)'].values[100:188]

	gauss1 = (1/(mu1*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu1)/sig)**2)

	gauss1_f = fft(gauss1)

	signal_f = fft(signalVolu)

	signal_f_wo_g1 = signal_f / gauss1_f.conj()

	signal_t_wo_g1 = ifft(signal_f_wo_g1)

	b, a = signal.butter(4, 0.05, 'low')
	
	zi = signal.lfilter_zi(b, a)
	
	z, _ = signal.lfilter(b, a, signal_t_wo_g1, zi=zi*signal_t_wo_g1[0])
	
	z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
	
	signal_t_wo_g1_filter = signal.filtfilt(b, a, signal_t_wo_g1)
		
	conc = (abs(max(z2))-coef_b)/coef_a ## mg/L
	
	conc_mmol_ml = conc/(785.02*1e3)  ## mol/L
	
	print(f'{filename}: {"{:.2f}".format(conc)} mg/L {"{:.2e}".format(conc_mmol_ml)} mmol/ml')



