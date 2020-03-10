import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sm
from mpl_toolkits.mplot3d import Axes3D

class hIFFL:
    def __init__(self, r_production, m_degradation, s_degradation, ms_degradation, c_production,
                 c_degradation, ci_production, ci_degradation, beta4, k_d):
        self.r_production = r_production
        self.m_degradation = m_degradation
        self.s_degradation = s_degradation
        self.ms_degradation = ms_degradation
        self.c_production = c_production
        self.c_degradation = c_degradation
        self.ci_production = ci_production
        self.ci_degradation = ci_degradation
        self.beta4 = beta4
        self.k_d = k_d

    def model(self, z, t, Temp):
        c, ci, ct, m, s = z[0], z[1], z[2], z[3], z[4]
        if Temp == 37: #or Temp == 36:
            dcdt = -c*t#-(self.c_production/self.c_degradation)*(1-np.exp(-self.c_degradation*t))
            dcidt = self.ci_production - self.ci_degradation*ci
        else:
            dcdt = self.c_production - self.c_degradation*c
            dcidt = -self.ci_degradation*ci
        dctdt = dcdt + dcidt
        dmdt = (self.r_production)/(1 + c/self.k_d) - self.beta4/(1+c/self.k_d)*m - self.m_degradation*m
        dsdt = (self.r_production)/(1 + c/self.k_d) - self.ms_degradation*(m**2/(m**2 + 1))*s - self.beta4/(1+c/self.k_d)*s  - self.s_degradation*s
        dzdt = [dcdt, dcidt, dctdt, dmdt, dsdt]
        return dzdt

    def modulation(self, t, period , off, on): #period * on or period * off must be less than len(t)
        modTemp = []
        for i in range(period):
            while len(t) > len(modTemp):
                for j in range(off):
                    modTemp.append(30)
                for k in range(31, 37):
                    modTemp.append(k)
                for l in range(on):
                    modTemp.append(37)
        return modTemp

    def runmodel(self, c, ci, ct, m, s, t, off, on):
        Temp = self.modulation(t, 3, off, on)
        z0 = [c, ci, ct, m, s]
        cr = np.empty_like(t)
        cir = np.empty_like(t)
        ctr = np.empty_like(t)
        mr = np.empty_like(t)
        sr = np.empty_like(t)
        cr[0], cir[0], ctr[0], mr[0], sr[0] = z0[0], z0[1], z0[2], z0[3], z0[4]
        for i in range(1, len(t)):
            tspan = [t[i-1],t[i]]
            z = odeint(self.model,z0,tspan, args =(Temp[i],))
            cr[i] = z[1][0]
            cir[i] = z[1][1]
            ctr[i] = z[1][2]
            mr[i] = z[1][3]#/(self.r_production/self.m_degradation)
            sr[i] = z[1][4]#/(self.r_production/(self.s_degradation + self.ms_degradation))
            z0 = z[1]
        return [cr.tolist(), cir.tolist(), ctr.tolist(), mr.tolist(), sr.tolist()]


if __name__ == '__main__':
    time = np.linspace(0, 60, 60)
    '''ht = hIFFL(0.45, 0.45, 0.05, 0.25, 0.45, 0.05, 0.5, 0.35, 0.65, .32)
    rr = ht.runmodel(0, 0, 0, 0, 0, time, 5, 1)
    plt.plot(time, rr[0])
    plt.xlabel('Time')
    plt.show()'''
    tv = 0.02
    while tv <= 1:
        ht = hIFFL(.85, 0.45, 0.5, tv, 0.05, 0.45, 0.35, 0.35, 0.65, .11)
        rr = ht.runmodel(0, 0, 0, 0, 0, time, 1, 1)
        plt.plot(rr[3], rr[4])
        tv += 0.02
    plt.title('Solution Trajectories')
    plt.xlabel('mf-Lon')
    plt.ylabel('m-Scarlet')
    plt.show()
    plt.plot(rr[4])
    plt.show()
