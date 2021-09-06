
mu1 = 0.75

mu2 = 0.6

sig = 0.05

x = df['E (V)']

gauss1 = (1/(mu1*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu1)/sig)**2)

#y_values = scipy.stats.norm(mu1, sig)

gauss2 = (1/(mu2*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu2)/sig)**2)

gaussT = gauss1 + gauss2

gaussT = signal.convolve(gauss1, gauss2)

deconv = scipy.signal.deconvolve( gaussT, gauss1 )

plt.plot(df['E (V)'], gauss1, label='gauss1')

plt.plot(df['E (V)'], gauss2, label='gauss2')

plt.plot(df['E (V)'], gaussT, label='gaussT')

plt.plot(df['E (V)'], deconv[1], label='deconv')

#plt.plot(df['E (V)'], y_values.pdf(df['E (V)']), label='py gass')

plt.legend()

plt.show()
