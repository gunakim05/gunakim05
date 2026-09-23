import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Loading Data
voltage_set, voltage_actual, current, voltage_range, current_range, uncertainty_voltage, uncertainty_current = np.loadtxt("Ohm Lab - 470Ohms.csv", delimiter=',', unpack=True, skiprows=1)


# Analyze Data - Fitting Linear Model
def linear_model(x, a, b):
    return a*x + b

# Fit Parameters
popt, pcov = curve_fit(linear_model, current, voltage_actual, sigma=uncertainty_voltage, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))

# Preidct Resistance in Ohms
resistance_predicted = popt[0] * 1000
resistance_predicted_uncertainty = pstd[0] * 1000

print("The predicted resistance is %.4g Ohm" %resistance_predicted, "+- %.1g Ohm" %resistance_predicted_uncertainty)

# Plot Data with Uncertainties
plt.errorbar(current, voltage_actual, yerr=uncertainty_voltage, marker='o', ls='', label='Data')

plt.plot(current, linear_model(current, *popt), label="Linear Fit")

# Decorate
plt.legend()
plt.title("Current vs Voltage for 470 Ohm Resistor")
plt.xlabel("Current (mA)")
plt.ylabel("Voltage (V)")
plt.show()