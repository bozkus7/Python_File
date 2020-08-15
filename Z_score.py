from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'matplotlib inline'


x = np.linspace(-4, 4, num = 100)
constant = 1.0 / np.sqrt(2*np.pi)
pdf_normal_distribution = constant * np.exp((-x**2) / 2.0)
fig, ax = plt.subplots(figsize=(10, 5));
ax.plot(x, pdf_normal_distribution);
ax.set_ylim(0);
ax.set_title('Normal Distribution', size = 20);
ax.set_ylabel('Probability Density', size = 20)

def normalProbabilityDensity(x):
    constant = 1.0 / np.sqrt(2*np.pi)
    return(constant * np.exp((-x**2) / 2.0) )

zoe_percentile, _ = quad(normalProbabilityDensity, np.NINF, 1.25)
mike_percentile, _ = quad(normalProbabilityDensity, np.NINF, 1.00)
print('Zoe: ', zoe_percentile)
print('Mike: ', mike_percentile)
