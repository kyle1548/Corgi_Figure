import numpy as np
class LinkLeg:
    def __init__(self) -> None:
        self.R = 0.1
        # self.r = 0.011 # no tire
        self.r = 0.0125 # no tire
        # self.r = 0.019    # with tire
        self.min_theta = np.deg2rad(17.0)
        self.max_theta = np.deg2rad(160.0)
        self.l1 = 0.8 * self.R  # l1
        self.l2 = self.R - self.l1  # l2
        self.l3 = 2.0 * self.R * np.sin(np.pi * 101.0 / 360.0)  # l3
        self.l4 = 0.88296634 * self.R   # l4
        self.l5 = 0.9 * self.R  # l5
        self.l6 = 0.4 * self.R  # l6
        self.l7 = 2.0 * self.R * np.sin(np.pi * 12.0 / 360.0)
        self.l8 = 2.0 * self.R * np.sin(np.pi * 50.0 / 360.0)
        
        self.l_AE = self.l5 + self.l6
        self.l_BF = 2.0 * self.R * np.sin(np.pi * 113.0 / 360.0)  
        self.l_BH = 2.0 * self.R * np.sin(np.pi * 17.0 / 360.0)
        self.to1 = np.pi * 39.5 / 180.0 # ang OBC at theta0
        self.to2 = - np.pi * 65.0 / 180.0
        self.tf = np.pi * 6.0 / 180.0
        self.th = np.pi * 121.0 / 180.0
        
        self.oe = 0  # = E (scalar)
        self.oe_d = 0
        self.oe_dd = 0

        self.db = 0 # l_BD
        self.db_d = 0
        self.db_dd = 0

        self.theta = 0
        self.theta_d = 0
        self.theta_dd = 0

        self.phi = 0    # ang_OEA
        self.phi_d = 0
        self.phi_dd = 0
        
        self.epsilon = 0    # ang_ y BD 
        self.epsilon_d = 0
        self.epsilon_dd = 0
        
        self.theta2 = 0 # ang_CBD
        self.theta2_d = 0
        self.theta2_dd = 0

        self.rho = 0    # ang_OGF
        self.rho_d = 0
        self.rho_dd = 0
        
    def calculate(self, theta, theta_d, theta_dd):
        self.theta = np.array(theta)
        self.theta_d = np.array(theta_d)
        self.theta_dd = np.array(theta_dd)

        # check theta range
        limit_u = self.theta > self.max_theta   # theta exceeding upper bound set to upper bound
        self.theta[limit_u] = self.max_theta
        limit_l = self.theta < self.min_theta   # theta below lower bound set to lower bound
        self.theta[limit_l] = self.min_theta
        if np.sum(limit_u) != 0:
            print("Limit upper bound")
        if np.sum(limit_l) != 0:
            print("Limit lower bound")
            
        # forward kinamatics
        self.phi = np.arcsin(self.l1 / self.l_AE *np.sin(self.theta) )    # ang_OEA
        self.D_phi()
        self.A = self.l1 * np.exp( 1j*(self.theta) )    # A
        self.B = self.R * np.exp(1j*(self.theta))   # B
        self.oe = self.l1 * np.cos(self.theta) - self.l_AE * np.cos(self.phi)
        self.E = self.l1 * np.cos(self.theta) - self.l_AE * np.cos(self.phi)
        self.D_oe()
        self.D = self.E + self.l6 * np.exp(1j*(self.phi))   # D
        db_2 = self.l5 * self.l5 + self.l2 * self.l2 - 2 * self.l5 * self.l2 * np.cos(np.pi - self.theta + self.phi)    # BD^2
        self.db = np.sqrt(db_2) # BD
        self.D_db()
        self.epsilon = np.arctan2(self.l5 * np.sin(self.phi) + self.l2 * np.sin(self.theta), self.l5 * np.cos(self.phi) + self.l2 * np.cos(self.theta)) 
        self.D_epsilon()
        self.theta2 = np.arccos((db_2 + self.l3 * self.l3 - self.l4 * self.l4) / (2.0 * self.db * self.l3)) 
        self.D_theta2()
        self.to = np.pi - self.theta2 + self.epsilon    # ang_ y BC + pi = ang_　-y B O1
        self.C = self.B + self.l3 * np.exp(1j*(self.to))
        self.O1 = self.B + self.R * np.exp(1j*(self.to + self.to1))
        self.D_O1()
        self.F = self.B + self.l_BF * np.exp(1j*(self.to + self.tf))
        self.H = self.B + self.l_BH * np.exp(1j*(self.to + self.th))
        self.rho = np.arcsin((self.R * np.sin(self.theta) + self.l_BF * np.sin(self.to + self.tf)) / self.l8)    # ang_OGF
        self.D_rho()
        self.G = self.F - self.l8 * np.exp(1j*(self.rho))
        self.D_G()
        self.O2 = self.G + self.R * np.exp(1j*(self.rho + self.to2))
        self.D_O2()
        self.symmetry()
        
    def D_phi(self):
        ph = self.l1 / self.l_AE * np.cos(self.theta) / np.cos(self.phi)
        self.phi_d = ph * self.theta_d
        self.phi_dd = ph *  self.theta_dd - self.l1 / self.l_AE * (np.sin(self.phi) * self.theta_d * self.theta_d + np.tan(self.phi) * np.cos(self.theta) * self.theta_d * self.phi_d) / np.cos(self.phi)
    
    def D_oe(self):
        self.oe_d = -self.l1 * np.sin(self.theta) * self.theta_d + self.l_AE * np.sin(self.phi) * self.phi_d
        self.oe_dd = -self.l1 * np.sin(self.theta) * self.theta_dd + self.l_AE * np.sin(self.phi) * self.phi_dd - self.l1 * np.cos(self.theta) * self.theta_d * self.theta_d + self.l_AE * np.cos(self.phi) * self.phi_d * self.phi_d
    
    def D_db(self):
        db2 = self.l5 * self.l2 * np.sin(np.pi - self.theta + self.phi)
        self.db_d = db2 * (-self.theta_d + self.phi_d) / self.db
        self.db_dd = db2  * (-self.theta_dd + self.phi_dd) / self.db + (-self.theta_d + self.phi_d) * (-self.theta_d + self.phi_d) / self.db - self.db_d  * (-self.theta_d + self.phi_d) * db2 / self.db / self.db
   
    def D_epsilon(self):
        self.epsilon_d = (self.l5 * self.l5 * self.phi_d + self.l2 * self.l2 * self.theta_d + self.l5 * self.l2 * np.cos(self.phi - self.theta) * (self.phi_d - self.theta_d)) / self.db / self.db
        self.epsilon_dd = (self.l5 * self.l5 * self.phi_dd + self.l2 * self.l2 * self.theta_dd + self.l5 * self.l2 * np.cos(self.phi - self.theta) * (self.phi_dd - self.theta_dd) - self.l5 * self.l2 * np.sin(self.phi - self.theta) * (self.phi * self.phi - self.theta * self.theta)) / self.db / self.db - 2 * (self.l5 * self.l5 * self.phi_d + self.l2 * self.l2 * self.theta_d + self.l5 * self.l2 * np.cos(self.phi - self.theta) * (self.phi_d + self.theta_d)) * self.db_d / self.db / self.db / self.db
    
    def D_theta2(self):
        self.theta2_d = -1 / np.sin(self.theta2) * (2 * self.db_d * (self.l4 * self.l4 - self.l3 * self.l3)) / 4 / self.db / self.db / self.l3
        self.theta2_dd = - (np.cos(self.theta2) * self.theta2_d) / np.sin(self.theta2) / np.sin(self.theta2) * (2 * self.db_d * (self.l4 * self.l4 - self.l3 * self.l3)) / 4 / self.db / self.db / self.l3 - 1 / np.sin(self.theta2) * (2 * (self.l4 * self.l4 - self.l3 * self.l3) * self.db_dd * (4 * self.db * self.db * self.l3) - 16 * self.db * self.db_d * self.db_d * self.l3 * (self.l4 * self.l4 - self.l3 * self.l3)) / 16 / self.l3 / self.l3 / (self.db * self.db * self.db * self.db)
    
    def D_rho(self):
        self.rho_d = 1 / np.cos(self.rho) * (self.R * np.cos(self.theta) * self.theta_d + self.l_BF * np.cos(np.pi - self.theta2 + self.epsilon + self.tf) * (-self.theta2_d + self.epsilon_d) ) / self.l8
        self.rho_dd = (self.R * self.theta_dd * np.cos(self.theta) - self.R * self.theta_d * self.theta_d * np.sin(self.theta) + self.l_BF * (-self.theta2_dd + self.epsilon_dd) * np.cos(np.pi - self.theta2 + self.epsilon + self.tf) - self.l_BF * (-self.theta2_d + self.epsilon_d) * (-self.theta2_d + self.epsilon_d) * np.sin(np.pi - self.theta2 + self.epsilon + self.tf)) / self.l8 / np.cos(self.rho) + np.sin(self.rho) * self.rho_d * (self.R * np.cos(self.theta) * self.theta_d + self.l_BF * np.cos(np.pi - self.theta2 + self.epsilon + self.tf) * (-self.theta2_d + self.epsilon_d)) / self.l8 / self.l8 / np.cos(self.rho) / np.cos(self.rho)
    
    def D_O1(self):
        self.O1_d = self.R * (self.theta_d) * np.exp(1j*(self.theta + np.pi / 2.0)) + self.R * (self.epsilon_d - self.theta2_d) * np.exp(1j*(self.to + self.to1 + np.pi / 2.0))
        self.O1_dd = self.R * (self.theta_dd) * np.exp(1j*(self.theta + np.pi / 2.0)) + self.R * (self.epsilon_dd - self.theta2_dd) * np.exp(1j*(self.to + self.to1 + np.pi / 2.0)) - self.R * (self.theta_d) * (self.theta_d) * np.exp(1j*(self.theta)) - self.R * (self.epsilon_d - self.theta2_d) * (self.epsilon_d - self.theta2_d) * np.exp(1j*(self.to + self.to1))
    
    def D_G(self):
        self.G_d = self.R * self.theta_d * np.exp(1j*(self.theta + np.pi / 2.0)) + self.l_BF * (self.epsilon_d - self.theta2_d) * np.exp(1j*(self.to + self.tf + np.pi / 2.0)) - self.l8 * self.rho_d * np.exp(1j*(self.rho + np.pi / 2.0))
        self.G_dd = self.R * self.theta_dd * np.exp(1j*(self.theta + np.pi / 2.0)) + self.l_BF * (self.epsilon_dd - self.theta2_dd) * np.exp(1j*(self.to + self.tf + np.pi / 2.0)) - self.l8 * self.rho_dd * np.exp(1j*(self.rho + np.pi / 2.0)) - self.R * self.theta_d * self.theta_d * np.exp(1j*(self.theta )) - self.l_BF * (self.epsilon_d - self.theta2_d) * (self.epsilon_d - self.theta2_d) * np.exp(1j*(self.to + self.tf )) + self.l8 * self.rho_d * self.rho_d * np.exp(1j*(self.rho ))
    
    def D_O2(self):
        self.O2_d = self.G_d + self.R * self.rho_d * np.exp(1j*(self.rho + self.to2 + np.pi / 2.0))
        self.O2_dd = self.G_dd + self.R * self.rho_dd * np.exp(1j*(self.rho + self.to2 + np.pi / 2.0)) - self.R * self.rho_d * self.rho_d * np.exp(1j*(self.rho + self.to2))
    
    def symmetry(self):
        self.O2_c = np.conjugate(self.O2)
        self.O2_d_c = np.conjugate(self.O2_d)
        self.O2_dd_c = np.conjugate(self.O2_dd)
        self.O1_c = np.conjugate(self.O1)
        self.O1_d_c = np.conjugate(self.O1_d)
        self.O1_dd_c = np.conjugate(self.O1_dd)
        self.F_c = np.conjugate(self.F)
        self.H_c = np.conjugate(self.H)
        self.A_c = np.conjugate(self.A)
        self.B_c = np.conjugate(self.B)
        self.C_c = np.conjugate(self.C)
        self.D_c = np.conjugate(self.D)
        self.O1_w = -self.theta2_d + self.epsilon_d
        self.O1_w_c = -self.O1_w
        self.O2_w = self.rho_d
        self.O2_w_c = -self.O2_w