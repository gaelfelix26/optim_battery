# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 15:48:19 2025

@author: gaelf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Temps de simulation
t = np.linspace(0, 800, 1500)

# Profil de courant (ex: 4 A pendant 500s)
def current(ti):
    if ti < 100:
        return 0.0
    elif ti < 450:
        return 2.0
    else:
        return 0.0

# Paramètres expérimentaux
OCV = [23.4755, 23.668, 23.8086, 24.0782, 24.7887]
SOCfull = [0.8352, 0.8477, 0.8646, 0.8971, 1]
SOCp = [0.8352, 0.8477, 0.8646, 0.8971]
R0 = [0.7605, 0.5778, 0.5390, 0.4604]
R1 = [1.4980, 1.6295, 1.5566, 1.4642]
R2 = [0.8096, 0.6465, 0.6522, 0.7014]
C1 = [4.1501, 5.4495, 7.8161, 5.1425]
C2 = [123.8446, 252.1806, 295.7867, 187.5892]

# Interpolateurs
voc_interp = interp1d(SOCfull, OCV, fill_value="extrapolate")
R0_interp = interp1d(SOCp, R0, fill_value="extrapolate")
R1_interp = interp1d(SOCp, R1, fill_value="extrapolate")
R2_interp = interp1d(SOCp, R2, fill_value="extrapolate")
C1_interp = interp1d(SOCp, C1, fill_value="extrapolate")
C2_interp = interp1d(SOCp, C2, fill_value="extrapolate")

param = {
    'Cbat': 2 * 3600,
    'OCVsoc': voc_interp,
    'R0soc': R0_interp,
    'R1soc': R1_interp,
    'R2soc': R2_interp,
    'C1soc': C1_interp,
    'C2soc': C2_interp,
    'Cth': 220,
    'Rth': 11,
    'T_ext': 298
}

# Fonction modèle
y0 = [1.0, 0.0, 0.0, 298.0]  # SOC, Vrc1, Vrc2, Température
def battery(t,y, cur_prof,  params):
    SOC, Vrc1, Vrc2, T = y
    SOC_interp = np.clip(SOC, 0, 0.92)
    I = cur_prof(t)

    # Accès aux paramètres via le dictionnaire
    Cbat = params['Cbat']
    R0soc = params['R0soc']
    R1soc = params['R1soc']
    R2soc = params['R2soc']
    C1soc = params['C1soc']
    C2soc = params['C2soc']
    Cth = params['Cth']
    Rth = params['Rth']
    T_ext = params['T_ext']

    dSOCdt = -I / Cbat
    dVrc1dt = (-Vrc1 / (R1soc(SOC_interp) * C1soc(SOC_interp))) + I / C1soc(SOC_interp)
    dVrc2dt = (-Vrc2 / (R2soc(SOC_interp) * C2soc(SOC_interp))) + I / C2soc(SOC_interp)

    R_eq = R0soc(SOC_interp) + R1soc(SOC_interp) + R2soc(SOC_interp)
    Q_joule = I ** 2 * R_eq
    Q_th = Q_joule + I * T * 0.01 * SOC
    dTdt = (Q_th - (T - T_ext) / Rth) / Cth

    return [dSOCdt, dVrc1dt, dVrc2dt, dTdt]

def battery_optim(t, y, cur_prof, params):
    SOC, Vrc1, Vrc2, T = y
    I = cur_prof(t)

    # Accès aux paramètres via le dictionnaire
    Cbat = params['Cbat']
    R0 = params['R0']
    R1 = params['R1']
    R2 = params['R2']
    C1 = params['C1']
    C2 = params['C2']
    Cth = params['Cth']
    Rth = params['Rth']
    T_ext = params['T_ext']

    dSOCdt = -I / Cbat
    dVrc1dt = (-Vrc1 / (R1 * C1)) + I / C1
    dVrc2dt = (-Vrc2 / (R2 * C2)) + I / C2

    R_eq = R0 + R1 + R2 
    Q_joule = I ** 2 * R_eq
    Q_th = Q_joule + I * T * 0.01 * SOC
    dTdt = (Q_th - (T - T_ext) / Rth) / Cth

    return [dSOCdt, dVrc1dt, dVrc2dt, dTdt]
""" pour charque boucle de sim_ batterie(), la fct aura besoin des valeurs 
de la boicle précedente à rentrer en condition initales de la boucle précedente
ce qui veut dire que la simu doit commencé relaxé, soit Vrc1=Vrc2=0V"""

def sim_batterie(t, y, cur_prof, params):
    sol = solve_ivp(
        lambda t, y: battery(t, y, cur_prof, params),
        [t[0], t[-1]],
        y,
        t_eval=t,
        method="RK45",
        max_step=1.0,
        rtol=1e-6,
        atol=1e-8
    )

    # Résultats
    SOC, Vrc1, Vrc2, T = sol.y
    SOC_clamped = np.clip(SOC, 0.8352, 0.8971)
    VOC = voc_interp(np.clip(SOC, 0.8352, 1.0))  # clamp uniquement dans les bornes de VOC
    I_vect = np.array([cur_prof(ti) for ti in t])
    R0_vect = R0_interp(SOC_clamped)
    V_bat = VOC - (Vrc1 + Vrc2 + I_vect * R0_vect)
    return V_bat, Vrc1, Vrc2,SOC,T

def sim_batterie_optim(t, cur_prof, y, params):
      
    """
    t = tableau de temps
    cur_prof = fonction ou interp du courant // t
    y = conditons initial [soc actuel, Vrc1 actuel, Vrc2 actuel, temperature actuelle]
    param = dictionnaire des paramètres meme format que pour batterie() """
    
    sol = solve_ivp(
        lambda t, y_: battery_optim(t, y_, cur_prof, params),
        [t[0], t[-1]],
        y,
        t_eval=t,
        method="RK45",
        max_step=1.0,
        rtol=1e-6,
        atol=1e-8
    )

    # Résultats
    SOC, Vrc1, Vrc2, T = sol.y
    VOC = voc_interp(np.clip(SOC, SOC[0], SOC[-1]))  # clamp uniquement dans les bornes de VOC
    I_vect = np.array([cur_prof(ti) for ti in t])
    R0 = params['R0']
    V_bat = VOC - (Vrc1 + Vrc2 + I_vect * R0)
    return V_bat, Vrc1[-1], Vrc2[-1],SOC[-1],T[-1]


# Simulation
sol = solve_ivp(
    lambda t, y: battery(t,  y,current, param),
    [t[0], t[-1]],
    y0,
    t_eval=t,
    method="RK45",
    max_step=1.0,
    rtol=1e-6,
    atol=1e-8
)

# Résultats
SOC, Vrc1, Vrc2, T = sol.y
SOC_clamped = np.clip(SOC, 0.8352, 0.8971)
VOC = voc_interp(np.clip(SOC, 0.8352, 1.0))  # clamp uniquement dans les bornes de VOC
I_vect = np.array([current(ti) for ti in t])
R0_vect = R0_interp(SOC_clamped)
V_bat = VOC - (Vrc1 + Vrc2 + I_vect * R0_vect)

# Affichage



plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(t, SOC)
plt.ylabel('SOC')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(t, V_bat)
plt.ylabel('Tension pack (V)')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(t, T - 273.15)
plt.ylabel('Température (°C)')
plt.xlabel('Temps (s)')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(t, Vrc1, label='Vrc1')
plt.plot(t, Vrc2, label='Vrc2')
plt.ylabel('Vrc (V)')
plt.xlabel('Temps (s)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(t, I_vect * R0_vect)
plt.ylabel('Tension R0 (V)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(t, I_vect)
plt.title("Profil de courant")
plt.xlabel("Temps (s)")
plt.ylabel("Courant (A)")
plt.grid(True)
plt.tight_layout()

plt.tight_layout()
plt.show()
