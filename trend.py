
import scipy

from scipy.integrate import quad

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

df = pd.read_csv('rota.csv')

def find_nearest(array, value):
     array = np.asarray(array)
     idx = (np.abs(array - value)).argmin()
     return array[idx]

#vvYs = ['y70','y40']
vvYs = ['y70', 'y60', 'y50', 'y40', 'y30', 'y20', 'y10']


for vvY in vvYs:
	X1 = 0.4
	X2 = 0.85

	pontoX1 = df[find_nearest(df['x'].values, X1) == df['x']]['x'].values[0]
	pontoY1 = df[find_nearest(df['x'].values, X1) == df['x']][vvY].values[0]

	pontoX2 = df[find_nearest(df['x'].values, X2) == df['x']]['x'].values[0]
	pontoY2 = df[find_nearest(df['x'].values, X2) == df['x']][vvY].values[0]

	a, b = np.polyfit([pontoX1, pontoX2],[pontoY1, pontoY2], 1)

	y_reta = a * df['x'].values + b

	plt.plot(df['x'].values, y_reta)

	plt.plot(df['x'].values, df[vvY].values, label=f'{vvY}')

	ini = df[find_nearest(df['x'].values, X1) == df['x']].index[0]

	fin = df[find_nearest(df['x'].values, X2) == df['x']].index[0]

	plt.fill_between(df['x'].values[ini:fin], y_reta[ini:fin],  df[vvY].values[ini:fin], color="grey", alpha=0.3, hatch='|')


	def eqReta(x,a,b):
	  return a * x + b

	def eqSignal(X2, df):
	  return df[find_nearest(df['x'].values, X2) == df['x']][vvY].values[0]

	I   = quad(eqReta, pontoX1, pontoX2, args=(a,b))

	Isi = quad(eqSignal, pontoX1, pontoX2, args=(df))

	print( vvY, Isi[0] - I[0])

plt.legend()

plt.savefig('areentreCurvas.png')
plt.show()
