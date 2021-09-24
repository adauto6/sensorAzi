import pandas as pd

import matplotlib.pyplot as plt

from scipy.fftpack import ifft, fft 

from scipy import signal

import numpy as np

from sklearn.metrics import r2_score


# filenames = AguaRIo.csv  analiseAguaLagoa.csv  analise.csv  poco.csv  torneira.csv
# analise.csv --> para obter a equacao da reta 

filenames = ['AguaRIo', 'analiseAguaLagoa',  'analise',  'poco',  'torneira']

for filename in filenames:
	print(filename)
#filename = 'analise'

	for i in range(1,5):
		plt.figure(i)
		plt.clf()
		
	df = pd.read_csv(filename+'.csv')

	#mu1 = 0.75

	mu1 = 0.6

	#mu1 = 0.23

	sig = 0.05

	#volumes = [1, 2, 5, 6, 7] 

	if filename == 'analise':
		volumes = [10, 20, 30, 40, 50, 60, 70]
		x_value = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] # Concentracao Retirada do volume e qtd de azitromicina colocada ( Calibracao )
		ptitle = 'DPV Au bicarbonato'
	else:
		volumes = [0, 30, 40, 50, 60, 70] # Agua de Rio
		x_value = [0.6, 0.8, 1.0, 1.2, 1.4] # Agua de Rio
		if filename == 'AguaRIo':
			ptitle = 'Am. de Agua de Rio'
		elif filename == 'analiseAguaLagoa':
			ptitle = 'Am. de Agua de Lagoa'
		elif filename == 'poco':
			ptitle = 'Am. de Agua de Poco'
		elif filename == 'torneira':
			ptitle = 'Am. de Agua de Torneira'	
	y_value = [] 
	vol0 = 1
	for volume in volumes:
		print(volume)
		if volume == 0:
			vol0 = df[f'Av. I2 {volume}uL'].values[100:188]
		else:
			signalVolu = df[f'Av. I2 {volume}uL'].values[100:188]/vol0

			x = df['E (V)'].values[100:188]

			plt.figure(1)

			plt.title(f'{ptitle}')
					
			plt.plot(x, signalVolu, label=f'Deconv{volume}uL - dividido por volume 0 uL')

			y_value.append(max(signalVolu))
			
			plt.grid()

			plt.xlabel('Potencial (V)')
			plt.ylabel(f'Corrente {volume}uL / Corrente 0uL')

			plt.legend()
	plt.savefig(f'divisao-{filename}.png', dpi=1200)
	plt.figure(2)

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
		
#	plt.plot(np.array(x_value)/(785.02*1e3), y_value, label='Dados')
#	plt.plot(np.array(x_value)/(785.02*1e3), y_calc,  label=f'y={"{:.2e}".format(abs(coef[0]))}*x + {"{:.2e}".format(abs(coef[1]))}, R^2 = {"{:.3f}".format(R2)}')


	plt.plot(np.array(x_value), y_value, label='Dados')
	plt.plot(np.array(x_value), y_calc,  label=f'y={"{:.2e}".format(np.real(coef[0]))}*x + {"{:.2e}".format(np.real(coef[1]))}, R^2 = {"{:.3f}".format(R2)}')

	concCalc = (-np.real(coef[1]))/np.real(coef[0])
	plt.xlabel('Concetracao (mg/L)')
	plt.ylabel('Corrente (apos divisao)')
	plt.legend()
	plt.title(f'{ptitle}, Conc. 0uL = {"{:.3f}".format(np.real(concCalc))} mg/L')
	plt.savefig(f'fitting-divisao{filename}.png')
	plt.clf()

		
