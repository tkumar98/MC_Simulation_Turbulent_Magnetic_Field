''' Monte Carlo Simulation to Calculate Probabilities due to turbulent field '''


import math
import numpy as np
import matplotlib.pyplot as plt
import random




########################################
#            Information               #
########################################
'''
1) Axion beam is propagating along z
2) Total distance travelled by axion beam = 1kpc = 1000 pc
3) B_reg is along x
4) B_turbulent has both x and y components
5) Using cell model of turbulent field with cell length L_cell = 20 pc
'''


########################################
#            Conversion Factors        #
########################################
G_GeV = 1.95e-20 # GeV^2 (Gauss->GeV^2)
pc_GeV = 1.56e32 # Gev^-1 (pc->GeV^-1)
m_GeV = 5.06e15 # GeV^-1 (meter->GeV^-1)
s_GeV = 1.52e24 # GeV^-1 (sec->GeV^-1)
cm_GeV = 5.06e13 # GeV^-1 (cm->GeV^-1)
pcinv_GeV = 1/pc_GeV # GeV (pc-inverse -> GeV)


########################################
#        Physical Parameters           #
#######################################
w = 50.0  # keV   # Energy of axion
m_a = float(1e-10) # eV
g_agamma = float(1e-11) # GeV^-1
B_reg = 3.e-6 # G
B_rms = 1.e-6 # G
L_cell = 20.0 # pc
L_corr = L_cell / 2.0 # pc
n_e = 1.e-6 # cm^-3
R = 1000.0 #pc

########################################
#        Integrator(Trapezoidal)       #
########################################
def integrate(f,a,b,h):
  l = np.arange(a,b,h)
  s = 0.0
  for i in range(len(l)-1):
    s = s + f(l[i]) + f(l[i+1])
  s = s*h/2
  return s


########################################
#       Oscillation Parameters         #
########################################
''' All in pc^-1 '''
def delta_pl(w,n_e):
  '''w in keV, n_e in cm^-3'''
  c = -1.1e-8 * (n_e/1e-7)*(1/w)
  return c

def delta_a(w,m_a):
  '''w in kev, m_a in eV'''
  c = -7.8e-1 * ((m_a/1e-10)**2)*(1/w)
  return c

def delta_m(g,B):
  '''g in GeV^-1 and B in G'''
  c = 1.52e-4 * (g/1e-10)*(B/1e-6)
  return c

def delta_osc(w,g,n_e,m_a,B):
  c = math.sqrt((delta_pl(w,n_e) - delta_a(w,m_a))**2 + 4*(delta_m(g,B))**2)
  return c


########################################
#       Random Field Generation        #
########################################
B_x_list = 0
B_y_list = 0
'''
random.seed()
B_x_list = np.random.default_rng().normal(0.0,1.0,(int(R/L_cell),))
B_y_list = np.random.default_rng().normal(0.0,1.0,(int(R/L_cell),))
'''
def B_x(zeta):
    zeta1 = math.floor(zeta/20)
    c = B_x_list[zeta1]*B_rms
    return c

def B_y(zeta):
    zeta1 = math.floor(zeta/20)
    c = B_y_list[zeta1]*B_rms
    return c


########################################
#             Probabilities            #
########################################
''' Conversion Probability due to regular field '''
def P_0(z):
    c1 = (g_agamma*B_reg*G_GeV/(delta_osc(w,g_agamma,n_e,m_a,B_reg)*pcinv_GeV))**2
    c2 = (math.sin(delta_osc(w,g_agamma,n_e,m_a,B_reg)*z/2))**2
    c = c1*c2
    return c


''' Conversion Probaility due to x component of turbulent field '''
def T_x(z):
  tx_re = lambda x1: G_GeV*B_x(x1)*math.cos(delta_osc(w,g_agamma,n_e,m_a,B_reg)*(x1-(z)/2))       # Real part of T_x
  tx_im = lambda x2: G_GeV*B_x(x2)*math.sin(delta_osc(w,g_agamma,n_e,m_a,B_reg)*(x2-(z)/2))       # Imaginary part of T_x

  c = 0.5*g_agamma*integrate(tx_re,0,(z),0.01)/(pcinv_GeV)
  s = 0.5*g_agamma*integrate(tx_im,0,(z),0.01)/(pcinv_GeV)
  return (c,s)

''' Conversion Probaility due to y component of turbulent field '''
def T_y(z):
  ty_re = lambda x1: G_GeV*B_y(x1)*math.cos(delta_osc(w,g_agamma,n_e,m_a,B_reg)*(x1-(z)/2))       # Real part of T_y
  ty_im = lambda x2: G_GeV*B_y(x2)*math.sin(delta_osc(w,g_agamma,n_e,m_a,B_reg)*(x2-(z)/2))       # Imaginary part of T_y
  
  c = 0.5*g_agamma*integrate(ty_re,0,(z),0.01)/(pcinv_GeV)
  s = 0.5*g_agamma*integrate(ty_im,0,(z),0.01)/(pcinv_GeV)
  return (c,s)

''' Total conversion probability '''
def P_agamma(z):
    p0 = P_0(z)
    tx = T_x(z)
    ty = T_y(z)
    px = tx[0]**2 + tx[1]**2
    py = ty[0]**2 + ty[1]**2
    c = p0 + px + py + 2*math.sqrt(p0)*ty[0]
    
    return c


########################################
#           Driver Function            #
########################################
def main():
    global B_x_list
    global B_y_list
    
    N = 10              # No. of samples
    
    f = open("Probabilities.txt",'w')
    
    z = np.arange(0,R,0.5) # in pc
    P = np.zeros(len(z))
    P_2 = np.zeros(len(z))
    
    for i in range(N):
        random.seed()
        B_x_list = np.random.default_rng().normal(0.0,1.0,(int(R/L_cell),))
        B_y_list = np.random.default_rng().normal(0.0,1.0,(int(R/L_cell),))
        
        for j in range(len(z)):
            P[j] += P_agamma(z[j])/N
            P_2[j] += (P_agamma(z[j]))**2 / N
            
    for i in range(len(z)):
        f.write(str(z[i])+" "+str(P[i])+" "+str(P[i]-math.sqrt(P_2[i]-P[i]**2))+" "+str(P[i]-math.sqrt(P_2[i]-P[i]**2))+"\n")
    
    f.close()
    
    
if __name__=='__main__':
    main()

