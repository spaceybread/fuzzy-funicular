import numpy as np
import matplotlib.pyplot as plt

# Define the range of n
n = np.arange(0, 128)

# Define the vector z(n)
z_n = 0.5 * (np.exp(2j * np.pi * 7 * n / 128) + np.exp(-2j * np.pi * 7 * n / 128)) + \
      4 * 0.5 * (np.exp(2j * np.pi * 12 * n / 128) + np.exp(-2j * np.pi * 2 * n / 128))

a = 64 * np.exp(2j * np.pi * 7 * n / 128)
b = 64 * np.exp(-1 * 2j * np.pi * 121 * n / 128)
c = 256 * np.exp(2j * np.pi * 12 * n / 128)
d = 256 * np.exp(- 1 * 2j * np.pi * 116 * n / 128)

z_n = (a + b + c + d) * (1/128)

# Plot the real and imaginary parts of z(n)
#plt.figure(figsize=(10, 6))
#plt.subplot(3, 1, 1)
#plt.plot(n, np.real(z_n), label='Real part')
#plt.plot(n, np.imag(z_n), label='Imaginary part')
#plt.title('Real and Imaginary parts of z(n)')
#plt.xlabel('n')
#plt.ylabel('Value')
#plt.legend()

# Compute the FFT of z(n)
fft_z_n = np.fft.fft(np.real(z_n))

# Plot the magnitude of the FFT
#plt.subplot(3, 1, 2)

plt.scatter(n, np.abs(fft_z_n))
plt.title('Magnitude of the FFT of z(n)')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

#plt.subplot(3, 1, 3)
#plt.plot(np.abs(fft_z_n))
#plt.title('Magnitude of the FFT of z(n)')
#plt.xlabel('Frequency')
#plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
