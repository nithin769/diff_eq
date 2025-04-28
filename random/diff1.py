import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Parameters
T = 10         # Half-width of the rectangular kernel
t_min = 0.0001
t_max = 100
dt = 0.00001

# Time vector
t = np.arange(t_min, t_max, dt)

# Signal
f = np.log(t)

# Define separate kernel time vector from -T to T
h_t = np.arange(-T, T, dt)
h = np.ones_like(h_t) # Uniform rectangular kernel

# Perform convolution
y = convolve(f, h, mode='same') * dt  # dt scaling

# Analytical expression
y_ana = np.zeros_like(t)
valid_indices = t > T
y_ana[valid_indices] = (
    (t[valid_indices] + T) * np.log(t[valid_indices] + T)
    - (t[valid_indices] - T) * np.log(t[valid_indices] - T)
    - 2 * T
)

# Errors
error = np.abs(y - y_ana)
mse = np.mean(error**2)

# Plot
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(t, f, label=r'$f(t) = \log t$')
plt.title('Input Signal')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(4,1,2)
plt.plot(h_t, h, label='Rectangular Kernel')
plt.title('Kernel $h(t)$')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(4,1,3)
plt.plot(t, y, label='Numerical Convolution')
plt.title('Convolution Result')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(4,1,4)
plt.plot(t, y, label='Numerical', color='blue')
plt.plot(t, y_ana, '--', label='Analytical', color='red')
plt.title('Verification: Numerical vs Analytical')
plt.grid(True)
plt.legend(loc='upper right')

# Print errors
print(f"Mean squared error: {mse}")

plt.tight_layout()
plt.show()
