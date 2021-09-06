
mu1 = 0.2

mu2 = 0.6

sig = 0.05

gauss1 = (1/(mu1*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu1)/sig)**2)

gauss2 = (1/(mu2*np.sqrt(2*np.pi)))*np.exp((-0.5)*( (x-mu2)/sig)**2)

yf1 = fft(gauss1)

yf2 = fft(gauss2)

yf = yf1 * yf2.conj()

plt.figure(1)
plt.plot(2.0/int(N) * np.abs(yf1[0:int(N/2)]), label='FFT 2')
plt.plot(2.0/int(N) * np.abs(yf1[0:int(N/2)]), label='FFT 1')
plt.plot(2.0/int(N) *  np.abs(yf[0:int(N/2)]), label='FFT Conv')


plt.legend()

new_sig = ifft(yf)

plt.figure(2)

plt.plot(x,new_sig, label='Conv')

plt.legend()

plt.figure(3)
plt.plot(x,new_sig, label='Conv')

plt.plot(x, gauss1, label='gauss1')

plt.plot(x, gauss2, label='gauss2')


plt.legend()


### Deconvolution

decFT_gauss1 = yf / yf2.conj()

new_signal_gauss1 = ifft(decFT_gauss1)

plt.figure(4)
plt.plot(x,new_signal_gauss1, label='Deconv')

plt.plot(x, gauss1, label='gauss1')

#plt.plot(x, gauss2, label='gauss2')


plt.legend()



plt.show()


