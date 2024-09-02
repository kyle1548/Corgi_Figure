import numpy as np
from numpy.polynomial import Polynomial


def rot_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])


def forward_G(theta, beta):
    Gy_coef = [-0.08004453, -0.04301230, -0.10580528, 0.08884963, -0.03102704, -0.00111215, 0.00303494, -0.00046524]
    Gy_poly = Polynomial(Gy_coef)
    
    xy = rot_matrix(beta) @ np.array([0, Gy_poly(theta)])
    return xy


def forward_U_right(theta, beta):
    Ux_coef = [-0.00966947, 0.03326843, -0.00141728, -0.00296485, -0.00086883, -0.00075226, 0.00037025, -0.00002507]
    Uy_coef = [-0.00066889, 0.01477225, -0.04975358, 0.02978929, -0.01978266, 0.00452581, 0.00037316, -0.00016161]
    Ux_poly = Polynomial(Ux_coef)
    Uy_poly = Polynomial(Uy_coef)
    
    xy = rot_matrix(beta) @ np.array([Ux_poly(theta), Uy_poly(theta)])
    return xy


def forward_U_left(theta, beta):
    Ux_coef = [0.00966947, -0.03326843, 0.00141728, 0.00296485, 0.00086883, 0.00075226, -0.00037025, 0.00002507]
    Uy_coef = [-0.00066889, 0.01477225, -0.04975358, 0.02978929, -0.01978266, 0.00452581, 0.00037316, -0.00016161]
    Ux_poly = Polynomial(Ux_coef)
    Uy_poly = Polynomial(Uy_coef)
    
    xy = rot_matrix(beta) @ np.array([Ux_poly(theta), Uy_poly(theta)])
    return xy


def forward_L_right(theta, beta):
    Lx_coef = [0.00620568, -0.00537354, -0.06028360, 0.02548121, 0.00855457, -0.00870950, 0.00213481, -0.00015989]
    Ly_coef = [0.02047862, -0.04890008, -0.08046279, 0.04414598, -0.00771598, -0.00429725, 0.00207745, -0.00021897]
    Lx_poly = Polynomial(Lx_coef)
    Ly_poly = Polynomial(Ly_coef)
    
    xy = rot_matrix(beta) @ np.array([Lx_poly(theta), Ly_poly(theta)])
    return xy


def forward_L_left(theta, beta):
    Lx_coef = [-0.00620568, 0.00537354, 0.06028360, -0.02548121, -0.00855457, 0.00870950, -0.00213481, 0.00015989]
    Ly_coef = [0.02047862, -0.04890008, -0.08046279, 0.04414598, -0.00771598, -0.00429725, 0.00207745, -0.00021897]
    Lx_poly = Polynomial(Lx_coef)
    Ly_poly = Polynomial(Ly_coef)
    
    xy = rot_matrix(beta) @ np.array([Lx_poly(theta), Ly_poly(theta)])
    return xy


def inverse_G(x, y):
    inv_G_len_coef = [-5.95872654, -198.85096075, -2844.02150622, -23373.03774867, -113378.19319984, -325114.82971918, -511715.14213397, -342018.73019766]
    inv_G_len_poly = Polynomial(inv_G_len_coef)

    xy_len = -np.sqrt(x**2+y**2)
    xy_ang = np.arctan2(y, x)
        
    theta = inv_G_len_poly(xy_len)
    beta = xy_ang + np.pi/2
    
    tb = (theta, beta)
    
    return tb


def inverse_U_right(x, y):
    inv_U_len_coef = [0.29524065, 31.24197798, -211.52507575, -399.51355637, 27999.64314180, -261557.63653199, 1068004.37189850, -1657739.16281409]
    inv_U_len_poly = Polynomial(inv_U_len_coef)
    
    xy_len = np.sqrt(x**2+y**2)
    
    theta = inv_U_len_poly(xy_len)
    
    xy_forward = forward_U_right(theta, 0)
    
    beta = np.arctan2(y, x) - np.arctan2(xy_forward[1], xy_forward[0])
    
    tb = (theta, beta)
    
    return tb


def inverse_U_left(x, y):
    inv_U_len_coef = [0.29524065, 31.24197798, -211.52507575, -399.51355637, 27999.64314180, -261557.63653199, 1068004.37189850, -1657739.16281409]
    inv_U_len_poly = Polynomial(inv_U_len_coef)
    
    xy_len = np.sqrt(x**2+y**2)
    
    theta = inv_U_len_poly(xy_len)
    
    xy_forward = forward_U_left(theta, 0)
    
    beta = np.arctan2(y, x) - np.arctan2(xy_forward[1], xy_forward[0])
    
    tb = (theta, beta)
    
    return tb


def inverse_L_right(x, y):
    inv_L_len_coef = [0.29530589, 11.02405536, -72.41173243, 986.64839625, -8170.44814417, 38939.29079420, -98157.04235297, 101562.43592433]
    inv_L_len_poly = Polynomial(inv_L_len_coef)
    
    xy_len = np.sqrt(x**2+y**2)
    
    theta = inv_L_len_poly(xy_len)
    
    xy_forward = forward_L_right(theta, 0)
    
    beta = np.arctan2(y, x) - np.arctan2(xy_forward[1], xy_forward[0])
    
    tb = (theta, beta)
    
    return tb


def inverse_L_left(x, y):
    inv_L_len_coef = [0.29530589, 11.02405536, -72.41173243, 986.64839625, -8170.44814417, 38939.29079420, -98157.04235297, 101562.43592433]
    inv_L_len_poly = Polynomial(inv_L_len_coef)
    
    xy_len = np.sqrt(x**2+y**2)
    
    theta = inv_L_len_poly(xy_len)
    
    xy_forward = forward_L_left(theta, 0)
    
    beta = np.arctan2(y, x) - np.arctan2(xy_forward[1], xy_forward[0])
    
    tb = (theta, beta)
    
    return tb


if __name__ == '__main__':
    # forward kinematics
    theta = 30  # deg
    beta = 45  # deg

    theta = np.deg2rad(theta)
    beta = np.deg2rad(beta)
    
    G_xy = forward_G(theta, beta)
    print('G_xy =', G_xy)

    U_xy_right = forward_U_right(theta, beta)
    print('U_xy_right =', U_xy_right)
    
    U_xy_left = forward_U_left(theta, beta)
    print('U_xy_left =', U_xy_left)
    
    L_xy_right = forward_L_right(theta, beta)
    print('L_xy_right =', L_xy_right)
    
    L_xy_left = forward_L_left(theta, beta)
    print('L_xy_left =', L_xy_left)
    
    
    # inverse kinematics (validation)
    print(f'\ntb = ({theta}, {beta})')
    
    tb = inverse_G(G_xy[0], G_xy[1])
    print('G_tb =', tb)
    
    tb = inverse_U_right(U_xy_right[0], U_xy_right[1])
    print('U_tb_right =', tb)
    
    tb = inverse_U_left(U_xy_left[0], U_xy_left[1])
    print('U_tb_left =', tb)
    
    tb = inverse_L_right(L_xy_right[0], L_xy_right[1])
    print('L_tb_right =', tb)
    
    tb = inverse_L_left(L_xy_left[0], L_xy_left[1])
    print('L_tb_left =', tb)
    