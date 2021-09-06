df = pd.read_csv('analise.csv')

mu = 0.75

sig = 0.05

x = df['E (V)']

gaussBirc = (1/(mu*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu)/sig)**2)

signal = df['Av. I2 40uL']

gauss = 5e-6*gaussBirc

deconv = scipy.signal.deconvolve( signal, gaussBirc )

#n = len(signal)-len(gauss)+1
# so we need to expand it by 

#s = int((len(signal)-n)/2)
#on both sides.
#deconv_res = np.zeros(len(signal))
#deconv_res[s:len(signal)-s-1] = deconv
#deconv = deconv_res

plt.plot(df['E (V)'], df['Av. I2 30uL'], label='30uL')

plt.plot(df['E (V)'], deconv[1], label='deconv')

plt.plot(df['E (V)'], gauss, label='gauss')

plt.legend()

plt.show()

