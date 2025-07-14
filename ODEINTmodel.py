# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:52:10 2025

@author: gaelf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from tabulate import tabulate
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import model

#import re

data = []
slicer = []   # contiendra les blocs
bloc = []     # bloc temporaire
R0=[]
RfVector=[]
RsVector=[]
CsVector=[]
CfVector=[]
SOCTable=[]
VOCTable = []

#double time constant model of the battery
def double_exp(t, V_inf, A1, tau1, A2, tau2):
    return V_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def constant_interp(value):
    return interp1d([0, 1], [value, value], fill_value="extrapolate")

# %%



"""----------------------------treating data------------------------------"""

FullTimeSim=[]
dataload = np.loadtxt("osc00008.txt", comments="#")

timeOut = dataload [-1,0]-dataload [0,0]


#downsing the dataload by deleting redondence
N = len(dataload[:, 0])
for i in range(N-1):
    if dataload[i,3] != dataload[i+1,3] :
        data.append(dataload[i,:])
    
data = np.array(data)
FullTimeSim = data[:,0]-data[0,0]
#print(data[:,3])

N=len(data[:, 0])

#print(N)

#determination for I median for the calculation
vecti = []
for i in range (N):
    if data[i, 5] != 0:
        vecti.append(data[i, 4])
I = np.median(vecti, axis = 0) 
print("\nCurrent median=",I,"A\n")


# slicing
for i in range(N - 1):
    bloc.append(data[i, :])
    if data[i, 5] != data[i+1, 5]:      #if the current swich of state
        slicer.append(np.array(bloc))   # on ajoute un bloc fini
        bloc = []

slicer=slicer[1:]
VOCTable.append(slicer[0][0,3])

for idx, blouc in enumerate(slicer):
    j = 0
    Vstart = blouc[j, 3]
    for i in [0, 1]:
        j = j + 1
        while blouc[j, 3] == blouc[j+1, 3]:
            blouc = np.delete(blouc, j, axis=0)
    R0.append(np.abs(Vstart - blouc[j+1, 3]) / I)
    blouc = np.delete(blouc, j, axis=0)
    slicer[idx] = blouc
    
R0 = R0[::2]

# ajouter le dernier bloc s'il reste des lignes
bloc.append(data[N - 1, :])
slicer.append(np.array(bloc))
# %%


pumping = slicer[::2]
relaxation = slicer[1::2]


def RC_calc(tau, I, A):
    R = A/I
    C = tau/R
    return R,C

for i in range(len(relaxation)):
    seconds = relaxation[i][:, 0]
    ib_control = relaxation[i][:, 1]
    test_mode = relaxation[i][:, 2]
    measured_voltage = relaxation[i][:, 3]
    measured_current = relaxation[i][:, 4]
    ib_dem = relaxation[i][:, 5]
    current = relaxation[i][:, 6]
    soc = relaxation[i][:, 7]
    ta = relaxation[i][:, 8]
    tb = relaxation[i][:, 9]
    tc = relaxation[i][:, 10]
    td = relaxation[i][:, 11]      
    t = seconds - seconds [0] #delete time offset 
 #   print( measured_voltage[0]) 

    """--------------------------processing------------------------------"""

    #preprocessing
    V_inf_guess = measured_voltage[-1]
    A1_guess = measured_voltage[0] - V_inf_guess
    A2_guess = A1_guess / 2
    tau1_guess = 5
    tau2_guess = 200
    p0 = [V_inf_guess, A1_guess, tau1_guess, A2_guess, tau2_guess]

    #curve fiting for a first determination of the parameters
    popt, _ = curve_fit(double_exp, t, measured_voltage,p0,maxfev=10000)

    #resulting curv
    V_fit = double_exp(t, *popt)

    #calculation of the RC grid
    

    tau1 = popt[2]
    tau2 = popt[4]
    A1= np.abs(popt[1])
    A2= np.abs(popt[3])

    Rf, Cf = RC_calc(tau1,I,A1)
    Rs, Cs = RC_calc(tau2,I,A2)
    RsVector.append(Rs)
    RfVector.append(Rf)
    CsVector.append(Cs)
    CfVector.append(Cf)
    SOCTable.append(soc[-1])
    VOCTable.append(measured_voltage[-1])

    #formating the display of the parameters
    param = [
        ["Amplitude",A1 ,A2,"V"],
        ["Time constant", tau1, tau2,"s"],
        ["Resistance", Rf, Rs,"Ω"],
        ["Capacitence", Cf, Cs,"F"]
        ]
    headers = ["","Fast", "Slow", "Units"]
    
    text = "Relaxation"
    print(f"{text:^10} : SOC=",soc[0]*100,"%     R0=", R0[i])
    print(tabulate(param, headers=headers, tablefmt="grid"))
    print("\n")
    
    #print(param)
    plt.figure(figsize=(10, 5))
    plt.plot(t, measured_voltage, label="Mesure", color="blue")
    plt.plot(t, V_fit, label="curve fiting", linestyle="--", color="red")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.title(f"Relaxation at SOC={soc[0] * 100:.1f}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #print(param)

# %%
"----------------------------------register---------------------------------"

# %%
"------------------------------model simulation-----------------------------"




# %%"
"--------------------------------optimisation--------------------------------"
Rfinit = RfVector
Rsinit = RsVector
Cfinit = CfVector
Csinit = CsVector
R0init = R0

def optim_model(x, y, time, cur_prof, param, MV):
    Rfoptim, Rsoptim, Cfoptim, Csoptim, R0optim = x

    argu = {
        'Cbat': 50 * 3600,
        'R0': R0optim,
        'R1': Rfoptim,
        'R2': Rsoptim,
        'C1': Cfoptim,
        'C2': Csoptim,
        'Cth': 100,
        'Rth': 1.5,
        'T_ext': 298
    }

    Vsim = model.sim_batterie_optim(time, cur_prof, y, argu)[0]

    if param == 'objective':
        errsq = np.sum((Vsim - MV)**2)
        print(f"Erreur = {errsq:.4f}")
        return errsq

    elif param == 'constraints':
        return [
            Rfoptim * Cfoptim - 5,
            Rsoptim * Csoptim - 80
        ]


# Problem definition
# def optim_model(x,y,cur_prof,param,MV):
    
#     Rfoptim = x[0]
#     Rsoptim = x[1]
#     Cfoptim = x[2]
#     Csoptim = x[3]
#     R0optim = x[4]
    
#     argu = {
#         'Cbat': 50 * 3600,       # 50 Ah -> en As
#         'R0': R0optim,              # Ohms
#         'R1': Rfoptim,             # Ohms
#         'R2': Rsoptim,             # Ohms
#         'C1': Cfoptim,              # Farads
#         'C2': Csoptim,              # Farads
#         'Cth': 100,              # J/K
#         'Rth': 1.5,              # K/W
#         'T_ext': 298             # Température ambiante en K
#         }
    
#     errsq = 0
#     print("oui")
#     Vsim = model.sim_batterie_optim(t,cur_prof,y,argu)[0]
#     print("certes")
#     for i in Vsim :
#         errsq = np.sum(Vsim[i]-MV)**2

#     pd.set_option('display.max_rows', None)
#     if param == 'constraints':
#         contrainte = []
#         for i in range(len(RfVector)):
#             for j in range(len(CfVector)):
#                 contrainte.append(Rfoptim[i] * Cfoptim[j]-5) 
#                 contrainte.append(Rsoptim[i] * Csoptim[j]-80)
# #                contrainte.append(R0[i])
# #                contrainte.append(R0[i]-RfVector[i])
#  #               contrainte.append(R0[i]-RsVector[i])
#         return contrainte
#     elif (param=='objective'):
#         print(errsq)
#         return errsq

# # Initial point
# Rfinit = RfVector
# Rsinit = RsVector
# Cfinit = CfVector
# Csinit = CsVector
# R0init = R0


y0 = [1.0, 0.0, 0.0, 298.0]
for i in range(len(relaxation)):
    seconds = relaxation[i][:, 0]
    ib_control = relaxation[i][:, 1]
    test_mode = relaxation[i][:, 2]
    measured_voltage = relaxation[i][:, 3]
    measured_current = relaxation[i][:, 4]
    ib_dem = relaxation[i][:, 5]
    current = relaxation[i][:, 6]
    soc = relaxation[i][:, 7]
    ta = relaxation[i][:, 8]
    tb = relaxation[i][:, 9]
    tc = relaxation[i][:, 10]
    td = relaxation[i][:, 11]      
    t = seconds - seconds [0]
    
    
    InitPoint=[Rfinit[i], Rsinit[i], Cfinit[i], Csinit[i], R0init[i]]
    bounds = []
    for j in [Rfinit[i], Rsinit[i], Cfinit[i], Csinit[i]]:
        bounds += [(0.05 * j, 10 * j)]
    bounds+=[(0.2 * R0init[i], 10 * R0init[i])]

    print('Bounds =', bounds)
    # Resolution of the problem
    Currentfct = interp1d(relaxation[i][:,0]-relaxation[i][0,0], relaxation[i][:,6], fill_value="extrapolate")
    objectif = lambda x: optim_model(x,y0,relaxation[i][:,0]-relaxation[i][0,0],Currentfct,MV=relaxation[i][:,3],param ="objective")
    contrainte = lambda x: optim_model(x,y0,relaxation[i][:,0]-relaxation[i][0,0],Currentfct,MV=relaxation[i][:,3],param = "constraints")
    result = scipy.optimize.fmin_slsqp(
        func=objectif,
        x0=InitPoint,
        bounds=bounds,
        f_ieqcons=contrainte,
        iter=1000,
        iprint=2,
        acc=1,
    )
# %%


    tab = [
        ["Resistance", result[0], result[1],"Ω"],
        ["Capacitence", result[2], result[3],"F"],
        ["R0",result[4],"Ω"," "]
        ]
    headers = ["","Fast", "Slow", "Units"]
    
    text = "Relaxation"
    print(f"{text:^10} : SOC=",soc[0]*100,"%     R0=", R0[i])
    print(tabulate(tab, headers=headers, tablefmt="grid"))
    print("\n")
    
    
    #plot construction
    argum = {
        'Cbat': 50 * 3600,
        'R0soc': constant_interp(result[0]),
        'R1soc': constant_interp(result[1]),
        'R2soc': constant_interp(result[2]),
        'C1soc': constant_interp(result[3]),
        'C2soc': constant_interp(result[4]),
        'Cth': 100,
        'Rth': 1.5,
        'T_ext': 298
}
    
    measured_current_interp = interp1d(t, measured_current, fill_value="extrapolate")
    V, v1, v2, so, tem = model.sim_batterie(t, y0, measured_current_interp, argum)
    plt.figure(figsize=(10, 5))
    plt.plot(t, measured_voltage , label="exp", color="red")
    plt.plot(t, V, label="simulation", color="green")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Simulation compared to experimentation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    y0=[soc[0],0,0,td[0]]

# %%
"----------------------------------displaying--------------------------------"



# def plout(var):
#     plt.figure(figsize=(10, 5))
#     for i in var :
#         plt.plot(temps, df[var]-273.15, label=i)
#         plt.plot(FullTimeSim,data[:,11])
#         plt.ylabel(i)
#     plt.xlabel("Time [s]")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
# plout(["capatherm.T"])

# #plot construction
# plt.figure(figsize=(10, 5))
# plt.plot(temps, reelVoltage , label="exp", color="red")
# plt.plot(temps, SimVoltage, label="simulation", color="green")
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.title("Simulation compared to experimentation")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# #print(param)

# print(omc.sendExpression('getErrorString()'))

# # uncomment to display all the variables
# #print(variables,"\n")

# # uncomment to display the first values of the simulated voltage
# #print(SimVoltage.head())

# print("done")
