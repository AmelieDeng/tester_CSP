#!/usr/bin/python
# -*-coding:utf-8-*-

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import curve_fit
print('hello, jiayin!')

print('你好, 家茵!')
# data_iv = pd.read_csv('C:\\Users\\jiayi\\OneDrive\\桌面\\IVIVC项目\\IVIVC的整体构建\\最终构建\\ivrat.csv')
data_ab = pd.read_csv('C:\\Users\\jiayi\\OneDrive\\桌面\\IVIVC项目\\IVIVC的整体构建\\最终构建\\18ab.csv')
# data_iv = pd.read_csv('C:\\Users\\jiayi\\OneDrive\\桌面\\IVIVC项目\\IVIVC的整体构建\\最终构建\\ivrat.csv')  

data_iv = pd.read_csv('C:\\Users\\jiayi\\OneDrive\\桌面\\IVIVC项目\\IVIVC的整体构建\\最终构建\\ivrat_3par.csv')  

def cumulate(ndarray, coef):
    total = 0
    for x, c in zip(np.nditer(ndarray), coef):
        total = total * c + x
        yield total
# def odf_2order(data_iv, ka, k_12, k_21, k_10):
#     def two_compartment_ivgtt(var, t, ka, k_12, k_21, k_10):
#         X_c, X_p = var
#         return np.array([ka + k_21 * X_p - (k_12 + k_10) * X_c, k_12 * X_c - k_21 * X_p])
#     t, C_t = data_ab['t'].to_numpy(), data_ab['C_t'].to_numpy()

    # curve = odeint(two_compartment_ivgtt, (0.0, 0.0), t, args=(ka, k_12, k_21, k_10))
# plt.plot(t, curve[:, 0], '-o')
# plt.show()
def lr2(data_iv, data_ab):
    k_12, k_21, k_10 = 24*data_iv['k_12'].to_numpy()[0], 24*data_iv['k_21'].to_numpy()[0], 24*data_iv['k_10'].to_numpy()[0]
    t, C_t = data_ab['t'].to_numpy(), data_ab['C_t'].to_numpy()

    delta_t = t[1:] - t[:-1]
    delta_C = C_t[1:] - C_t[:-1]
    coef = np.exp(-k_21 * delta_t)
    X_p_V_c = np.array(list(cumulate(((k_12 * C_t[:-1] / k_21) * (1 - np.exp(-k_21 * delta_t)) + k_12 * delta_C * delta_t / 2), coef)))

    AUC_t = np.cumsum((C_t[:-1] + C_t[1:]) * (t[1:] - t[:-1]) / 2)
    AUC_inf = AUC_t[-1]

    F_a = (C_t[1:] + k_10 * AUC_t + X_p_V_c) / (k_10 * AUC_inf)

    F_a = pd.DataFrame(np.hstack((np.zeros(1), F_a)), columns=['F_a'])
    F_a['t'] = t

    log_y = np.log10(100*(1-F_a['F_a'])) - 2    
    

    def get_ka(x, a):
        return a*x
    popt, pcov = curve_fit(get_ka, t, log_y)
    y_pred = [get_ka(i, popt[0]) for i in t]
    ka = -popt[0] * 2.303
    print(k_10 * AUC_inf)
    return F_a, ka

def odf_2order(data_ab, ka, k_12, k_21, k_10):
    def two_compartment_ivgtt(var, t, ka, k_12, k_21, k_10):
        Xa, X_c, X_p = var
        return np.array([-ka * Xa, ka * Xa + k_21 * X_p - (k_12 + k_10) * X_c, k_12 * X_c - k_21 * X_p])
    t, C_t = data_ab['t'].to_numpy(), data_ab['C_t'].to_numpy()

    curve = odeint(two_compartment_ivgtt, (800, 0.0, 0.0), t, args=(ka, k_12, k_21, k_10))
    print(curve[:,:])
    r2 = r2_score(C_t,curve[:, 1])
    print('2order ode R2:', r2)
    
# try:
#     data_iv['k_12'], data_iv['k_21'], data_iv['k_10']
# except:
#     data_iv['k_12'], data_iv['k_21'], data_iv['k_10'] = get_pk_parameters_lr2(data_iv, n_last=4, n_first=3)

# F_a, ka = lr2(data_iv, data_ab)
# k0 = ka*24
# print(F_a)


# odf_2order(data_ab, ka, k_12, k_21, k_10)
# print(F_a)
# 40*1000ng
def recurrence_3compart(data_iv, data_ab, F_a):
    k_12, k_21, k_10= data_iv['k_12'].to_numpy()[0], data_iv['k_21'].to_numpy()[0], data_iv['k_10'].to_numpy()[0]
    # k_12, k_21, k_10 = 24*data_iv['k_12'].to_numpy()[0], 24*data_iv['k_21'].to_numpy()[0], 24*data_iv['k_10'].to_numpy()[0]
    # k_12, k_21, k_10, k_13, k_31 = 24*data_iv['k_12'].to_numpy()[0], 24*data_iv['k_21'].to_numpy()[0], 24*data_iv['k_10'].to_numpy()[0], 24*data_iv['k_13'].to_numpy()[0], 24*data_iv['k_31'].to_numpy()[0]
    t_raw, C_t_raw = data_ab['t'].to_numpy(), data_ab['C_t'].to_numpy()

    AUC_t = np.cumsum((C_t_raw[:-1] + C_t_raw[1:]) * (t_raw[1:] - t_raw[:-1]) / 2)

    AUC_inf = AUC_t[-1]
    X1, X2 = 0, 0
    X1_all = []
    for i in range(len(t_raw)-1):
        ti = t_raw[i:i+2]
        cti = F_a[i:i+2]
        fx = interp1d(ti, cti, kind='linear')
        xInterp = np.linspace(ti[0],ti[1],101)
        yInterp = fx(xInterp) 
        t = xInterp #含有100个点的
        C_t = yInterp
        for n in range(100):
            k12, k21, k10= k_12*24, k_21*24, k_10*24

            dt = t[n+1] - t[n]

            dXa = k10 * AUC_inf * (C_t[n+1]- C_t[n])

            dX2 = (k12*X1 - k21*X2)*dt

            dX1 = dXa - dX2 - k10*X1*dt

            a = X2
            X2 = a + dX2

            c = X1
            X1 = c + dX1
        # X1_all.append(X1)
        print(dXa)
    # X1_all = np.insert(X1_all,0,0)
    # X1_all = pd.DataFrame(X1_all, columns=['X1_all'])
    # X1_all['t'] = t_raw
    # print(X1_all)
    # log_y = np.log10(100*(1-F_a['F_a'])) - 2    
    # def get_ka(x, a):
    #     return a*x
    # popt, pcov = curve_fit(get_ka, t, log_y)
    # y_pred = [get_ka(i, popt[0]) for i in t]
    # ka = -popt[0] * 2.303

    

try:
    data_iv['k_12'], data_iv['k_21'], data_iv['k_10']
except:
    data_iv['k_12'], data_iv['k_21'], data_iv['k_10'] = get_pk_parameters_lr2(data_iv, n_last=4, n_first=3)

F_a, ka = lr2(data_iv, data_ab)
F_a= F_a['F_a'].to_numpy()
k_12, k_21, k_10= data_iv['k_12'].to_numpy()[0], data_iv['k_21'].to_numpy()[0], data_iv['k_10'].to_numpy()[0]
print(24*k_12, 24*k_21, 24*k_10)

# recurrence_3compart(data_iv, data_ab, F_a)