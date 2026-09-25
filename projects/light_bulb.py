import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Loading Data
voltage_set, voltage_actual, current, voltage_range, current_range, uncertainty_voltage, uncertainty_current = np.loadtxt("lightbulb_lab.csv", delimiter=',', unpack=True, skiprows=5)

def model_function(x, a, b):
    return a*((x**b))
    
def theory_function(x, a):
    return a*(x**(0.6))
    
# Fit Parameters - Theory Function
poptT, pcovT = curve_fit(
    theory_function, 
    voltage_actual, 
    current,
    sigma=uncertainty_current, 
    absolute_sigma=True
    )

pstdT = np.sqrt(np.diag(pcovT))

# Fit Parameters - Model Function
poptM, pcovM = curve_fit(
    model_function, 
    voltage_actual, 
    current,
    sigma=uncertainty_current, 
    absolute_sigma=True
    )

pstdM = np.sqrt(np.diag(pcovM))

print(*poptM)
print(*poptT)

# Predicted voltage
current_model = model_function(voltage_actual, *poptM)

current_theory = theory_function(voltage_actual, *poptT)

residual_model = current - current_model
residual_theory = current - current_theory

# Chi-squared
chi2_model = np.sum(
    ((current-current_model) / uncertainty_current)**2
    )

chi2_theory = np.sum(
    ((current-current_theory) / uncertainty_current)**2
    )

# Degrees of freedom
N_model = len(current)
m_model = len(poptM)
degrees_of_freedom_model = N_model - m_model

N_theory = len(current)
m_theory = len(poptT)
degrees_of_freedom_theory = N_theory - m_theory

# Reduced Chi-squared
reduced_chi2_model = chi2_model / degrees_of_freedom_model
reduced_chi2_theory = chi2_theory / degrees_of_freedom_theory

print("Chi-squared model= %.3f" % chi2_model)
print("Degrees of freedom model=", degrees_of_freedom_model)
print("Reduced chi-squared model= %.3f" %reduced_chi2_model)

print("Chi-squared theory= %.3f" % chi2_theory)
print("Degrees of freedom theory=", degrees_of_freedom_theory)
print("Reduced chi-squared theory= %.3f" %reduced_chi2_theory)

# Plot Data with Uncertainties
#plt.errorbar(voltage_actual, current, yerr=uncertainty_current, marker='o', ls='', label='Data')
#plt.plot(voltage_actual, model_function(voltage_actual, *poptM), label='Model Function')
#plt.plot(voltage_actual, theory_function(voltage_actual, *poptT), label = 'Theory Function')

# Residual
plt.errorbar(voltage_actual, residual_model, yerr=uncertainty_current, marker='o', ls='', label = "Residual Model")
plt.errorbar(voltage_actual, residual_theory, yerr=uncertainty_current, marker='o', ls='', label = "Reisdual Theory")

# Decorate
plt.legend()
plt.xlabel("Voltage (V)") 
plt.ylabel("Current (mA)")
#plt.loglog() 
plt.show()
