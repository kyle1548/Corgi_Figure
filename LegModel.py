import numpy as np
import LinkLeg

class Leg:
    def __init__(self, offset) -> None:
        self.linkleg = LinkLeg.LinkLeg()
        self.offset = offset
        self.beta0 = np.pi/2
        
    def calculate(self, theta, theta_d, theta_dd, beta, beta_d, beta_dd):
        self.beta = beta
        self.beta_d = beta_d
        self.beta_dd = beta_dd
        rot_ang =  np.exp( 1j*(np.array(beta) + self.beta0) )
        rot_vel =  np.array(beta_d) * np.exp(1j*(np.array(beta) + np.pi / 2.0))
        # rot_accel = - np.array(beta_dd) * np.exp(1j*np.array(beta))
        self.linkleg.calculate(theta, theta_d, theta_dd)
        self.linkleg.O1 = rot_ang * self.linkleg.O1
        self.linkleg.A = rot_ang * self.linkleg.A
        self.linkleg.A_c = rot_ang * self.linkleg.A_c
        self.linkleg.B = rot_ang * self.linkleg.B
        self.linkleg.B_c = rot_ang * self.linkleg.B_c
        self.linkleg.C = rot_ang * self.linkleg.C
        self.linkleg.C_c = rot_ang * self.linkleg.C_c
        self.linkleg.D = rot_ang * self.linkleg.D
        self.linkleg.D_c = rot_ang * self.linkleg.D_c
        self.linkleg.E = rot_ang * self.linkleg.E
        self.linkleg.O1_c = rot_ang * self.linkleg.O1_c
        self.linkleg.O2 = rot_ang * self.linkleg.O2
        self.linkleg.O2_c = rot_ang * self.linkleg.O2_c
        self.linkleg.O1_d = rot_ang * self.linkleg.O1_d + rot_vel * self.linkleg.O1
        self.linkleg.O1_d_c = rot_ang * self.linkleg.O1_d_c + rot_vel * self.linkleg.O1_c
        self.linkleg.O2_d = rot_ang * self.linkleg.O2_d + rot_vel * self.linkleg.O2
        self.linkleg.O2_d_c = rot_ang * self.linkleg.O2_d_c + rot_vel * self.linkleg.O2_c
        self.linkleg.G = rot_ang * self.linkleg.G
        self.linkleg.H = rot_ang * self.linkleg.H
        self.linkleg.F = rot_ang * self.linkleg.F
        self.linkleg.H_c = rot_ang * self.linkleg.H_c
        self.linkleg.F_c = rot_ang * self.linkleg.F_c
        self.linkleg.G_d = rot_ang * self.linkleg.G_d + rot_vel * self.linkleg.G
        # angular velocity
        self.linkleg.O1_w += beta_d
        self.linkleg.O1_w_c += beta_d
        self.linkleg.O2_w += beta_d
        self.linkleg.O2_w_c += beta_d