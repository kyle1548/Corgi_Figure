import numpy as np
import Contact_Map
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
from FittingCoefficient import *
import inv_move_kinematics
    
import sys
sys.path.append('corgi_mpc/src')
from corgi_mpc.src.swing import *

contact_map = Contact_Map.ContactMap()
linkleg = contact_map.leg.linkleg
r = linkleg.r
outer_radius = linkleg.R + r
radius = linkleg.R

#### Find theta/beta when a given point on the lower rim contact ground. ####
def lower_contact(alpha, samples=1): # rad, samples per degree
    vO2F = -1j*radius * np.exp(-1j*alpha)    # vector from O2 to F
    theta = np.deg2rad(np.arange(17, 160, 1/samples))
    beta0 = np.zeros(theta.shape)
    contact_map.mapping(theta, beta0)
    O2_b0 = linkleg.O2  # O2 position when beta=0
    F_b0 = linkleg.F    # F position when beta=0
    v_O2F_b0 = F_b0 - O2_b0   # vector from O2 to F ( beta=0 )
    rotate = np.angle(vO2F/v_O2F_b0)    # beta = angle bewteen two vectors
    contact_map.mapping(theta, rotate)
    center = linkleg.O2 # O2 position when alpha contact ground
    return theta, rotate, center   # theta, beta, center of contact rim

#### Find theta/beta when a given point on the upper rim contact ground. ####
def upper_contact(alpha, samples=1): # rad, samples per degree
    vO1H = -1j*radius * np.exp(-1j*alpha)    # vector from O1 to H
    theta = np.deg2rad(np.arange(17, 160, 1/samples))
    beta0 = np.zeros(theta.shape)
    contact_map.mapping(theta, beta0)
    O1_b0 = linkleg.O1  # O1 position when beta=0
    H_b0 = linkleg.H    # H position when beta=0 
    vO1H_b0 = H_b0 - O1_b0   # vector from O1 to H ( with beta=0 )
    rotate = np.angle(vO1H/vO1H_b0) # beta = angle bewteen two vectors
    contact_map.mapping(theta, rotate)
    center = linkleg.O1 # O1 position when alpha contact ground
    return theta, rotate, center   # theta, beta, center of contact rim


def hip_trajectory(d, x, W, H):
    phi = np.arctan(H/W)         # φ is a constant
    y = np.tan(phi)*x + d/np.cos(phi) + H   # y = a*x + d*(a^2+1)^0.5 + y(0)
    return y

def hip_trajectory_inv(phi, d, y):
    x = ( y - d/np.cos(phi) ) / np.tan(phi)   # x = ( y - d*(a^2+1)^0.5 ) / a
    return x

def find_hip_alpha0(phi, d, hip_coef, x):
    y = np.polyval(hip_coef, x) 
    return hip_trajectory(phi, d, x) - y

#### Hip position for start climbing stair ####
def get_initial_hip(alpha0, d, initial_foothold, guess=0, W=0.27, H=0.17):
    phi = np.arctan(H/W)         # φ is a constant
    theta, beta, center = lower_contact(alpha0)
    contact_points = np.array(center - 1j*outer_radius)
    x = contact_points.real
    y = contact_points.imag  
    contact_points = np.array([x, y])    # 2 *n_alpha *n_theta
    hip_p = initial_foothold - contact_points
    hip_coef = np.polyfit(hip_p[0], hip_p[1], 7)

    #### Solve ####
    equation_to_solve = lambda guess_hip: find_hip_alpha0(phi, d, hip_coef, guess_hip)
    solution =  fsolve(equation_to_solve, guess)   # initial guess
    hip0 = np.array([solution, hip_trajectory(phi, d, solution)])

    print("交点:", solution)

    # 绘制函数和交点
    x = np.linspace(min(hip_p[0]), max(hip_p[0]), 100)

    # stair
    n_stair = 3
    # Sx = [ W*j for i in range(-1, n_stair) for j in [i, i+1] ] + [W*n_stair]
    # Sy = [-H] + [ H*j for i in range(-1, n_stair) for j in [i, i+1] ]
    # plt.plot(Sx, Sy, 'b-', lw=1)
    plt.axis('equal')

    Sx = [ W*j for i in range(-1, n_stair) for j in [i, i+1] ] + [W*n_stair]
    Sy = [-H] + [ H*j for i in range(-1, n_stair) for j in [i, i+1] ]

    plt.plot(Sx, Sy, 'b-', lw=1)
    plt.plot([-1, -W], [-H, -H], 'b-', lw=1)
    plt.plot([W*n_stair, W*n_stair + 0.5], [H*n_stair, H*n_stair], 'b-', lw=1)
    # line of stair 
    # plt.plot(Sx[0::2], Sy[0::2], 'r-')
    # plt.plot(Sx[0::2], Sy[0::2] + np.array(d*CL/W), 'r-')

    plt.plot(x, hip_trajectory(phi, d, x), label='hip_trajectory')
    plt.plot(x, np.polyval(hip_coef, x), label='hip_alpha0')
    plt.scatter(solution, hip_trajectory(phi, d, solution), color='red', zorder=5)  # 用红色点标注交点
    plt.legend()
    plt.title('intersection point')
    plt.grid(True)
    plt.show()
    
    return hip_coef, hip0

    
#### For fsolve find delta alpha (alpha-alpha0) for the given hip, foothold and hip coefficient of alpha0. ####
def find_d_alpha(hip, foothold, hip_coef, d_alpha, radius=outer_radius):   # next hip, current foothold, hip_coef, guess d_alpha \\\ hip_coef: coefficient of hip position when initial alpha of lower rim contacting ground
    angle = d_alpha[0]
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    displace = angle*radius  
    guess_p = foothold + [displace, radius] # guess_O2 = foothold + (displace, outer_radius) ( represent in world coordinate )
    guess_p2 = guess_p - hip   # guess_O2 represent in hip coordinate
    guess_point = guess_p + (rot_matrix @ -guess_p2.reshape(2, 1)).reshape(-1) - [displace, 0]    # point for this guess alpha
    point_y = np.polyval(hip_coef, guess_point[0])    # y point on the hip_alpha0 curve given x of guess_point
    return guess_point[1] - point_y

# def parabolic_blends(p, t, tp=0.2): # position, time: 0~1, acceleration time
#     p = np.array(p)
#     t = np.array(t)
#     vi = 0  # initial velocity
#     vf = 0  # final velocity
#     n_points = p.shape[0]
#     tp = tp * np.ones(n_points)
#     t[0] += 0.5 * tp[0]
#     t[-1] -= 0.5 * tp[-1]
#     parabolic_arr = np.zeros((2*n_points-1, 3))
#     v = np.hstack((vi, np.diff(p)/np.diff(t), vf)) 
#     a = np.diff(v)/tp

#     parabolic_arr[0] = np.array([0.5*a[0], vi, p[0]]) # 1st segment, 0~tp, acceleration
#     for i in range(n_points-1): 
#         parabolic_arr[2*i+1] = v[i+1]*np.array([0, 1, -t[i]]) + np.array([0, 0, p[i]])  # constant speed
#         tmp = t[i+1] - 0.5*tp[i+1] # acceleration start time
#         parabolic_arr[2*i+2] = parabolic_arr[2*i+1] + 0.5*a[i+1]*np.array([1, -2*tmp, tmp**2])  # acceleration
#     return parabolic_arr

def parabolic_blends(p, t, tp=0.2, vi=0, vf=0): # position, time: 0~1, acceleration time, initial velocity, final velocity
    p = np.array(p).astype(float)
    t = np.array(t)
    n_points = p.shape[0]
    tp = tp * np.ones(n_points)
    t[0] += 0.5 * tp[0]
    t[-1] -= 0.5 * tp[-1]
    p0 = p[0]
    if vf is None:
        vf = (np.diff(p)/np.diff(t))[-1]
    parabolic_arr = np.zeros((2*n_points-1, 3))
    v = np.hstack((vi, np.diff(p)/np.diff(t), vf)) 
    a = np.diff(v)/tp

    a0 = ((p[1]-v[0]*tp[0]-p[0])/(t[1]-tp[0]) - v[0]) / ( 2*tp[0] + (tp[0])**2/(t[1]-tp[0]) )   # p0(t) = a t^2 + v0 t + p0
    p[0] = np.polyval(np.array([a0, v[0], p[0]]), tp[0])    # p0(tp/2)
    t[0] = tp[0]    # t0 = tp/2
    v = np.hstack((vi, np.diff(p)/np.diff(t), vf)) 
    a = np.diff(v)/tp
    parabolic_arr[0] = np.array([0.5*a[0], v[0], p0]) # 1st segment, 0~tp, acceleration
    for i in range(n_points-1): 
        if i==0:
            parabolic_arr[2*i+1] = v[i+1]*np.array([0, 1, -t[i]]) + np.array([0, 0, p[i]])  # constant speed   
        else:
            parabolic_arr[2*i+1] = v[i+1]*np.array([0, 1, -t[i]]) + np.array([0, 0, p[i]])  # constant speed   
        tmp = t[i+1] - 0.5*tp[i+1] # acceleration start time
        parabolic_arr[2*i+2] = parabolic_arr[2*i+1] + 0.5*a[i+1]*np.array([1, -2*tmp, tmp**2])  # acceleration
    return parabolic_arr

def get_parabolic_point(p, parabolic_arr, t, tp=0.1):
    t = np.array(t)
    n_points = t.shape[0]
    tp = tp * np.ones(n_points)
    t[0] += 0.5 * tp[0]
    t[-1] -= 0.5 * tp[-1]
    
    segments = np.zeros(parabolic_arr.shape[0])
    segments[0] = t[0] + 0.5 * tp[0]
    for i in range(n_points-1): 
        segments[2*i+1] = t[i+1] - 0.5 * tp[i+1]
        segments[2*i+2] = t[i+1] + 0.5 * tp[i+1]
    
    if p < 0: 
        return np.polyval(parabolic_arr[0], p/p) 
    elif p >= 1.0:
        return np.polyval(parabolic_arr[-1], p/p) 
    else:
        for idx, segment in enumerate(segments):
            if p < segment:
                return np.polyval(parabolic_arr[idx], p) 

    print("ERROR IN get_parabolic_point")
    return 0

# def parabolic_blends(start, middle, end, tp=0.1):
#     vi = 0, vf = 0  # initial, final velocity
#     v_interval = np.array([0.5-tp/2, 1-tp/2-0.5])
#     a_interval = np.array([tp, tp*2, tp])
#     parabolic_arr = np.zeros((5, 3))
#     p = np.array([start, middle, end])  # 3
#     v = np.hstack((vi, np.diff(p)/v_interval, vi)) # 4
#     a = np.diff(v)/a_interval  # 3
    
#     parabolic_arr[0] = [a[0]/2, v[0], p[0]] # 1st segment, 0~tp
#     parabolic_arr[1] = [0, v[1], p[0]-v[1]*(tp/2)] # 2nd segment, tp~0.5-tp
#     parabolic_arr[2] = [a[1]/2, v[1]-a[1]*(0.5-tp), p[0]-(tp/2)*v[1]+a[1]/2*(0.5-tp)**2] # 3rd segment, 0.5-tp ~ 0.5+tp
#     parabolic_arr[3] = [0, v[2], p[1]-v[2]*(0.5)] # 4th segment, 0.5+tp ~ 1-tp
#     parabolic_arr[4] = [a[2]/2, v[2]-a[2]*(1-tp), p[1]-0.5*v[2]+a[2]/2*(1-tp)**2] # last segment, 1-tp ~ 1
#     return parabolic_arr


# def get_parabolic_point(ratio, parabolic_arr, tp=0.1):
#     if 0 <= ratio < tp:
#         return np.polyval(parabolic_arr[0], ratio) 
#     elif tp <= ratio < 0.5-tp:
#         return np.polyval(parabolic_arr[1], ratio) 
#     elif 0.5-tp <= ratio < 0.5+tp:
#         return np.polyval(parabolic_arr[2], ratio) 
#     elif 0.5+tp <= ratio < 1-tp:
#         return np.polyval(parabolic_arr[3], ratio)
#     elif 1-tp <= ratio < 1:
#         return np.polyval(parabolic_arr[4], ratio)  
#     elif ratio >= 1:
#         return np.polyval(parabolic_arr[4], ratio/ratio)
#     else:
#         print("ERROR")
#         return 0


def create_command_csv(theta_command, beta_command, file_name): # 4*n, 4*n
    # Tramsform beta, theta to right, left motor angles
    theta_0 = np.array([-17, 17])*np.pi/180
    theta_beta = np.array([theta_command, beta_command]).reshape(2, -1)   # 2*(4*n)
    phi_r, phi_l = np.array([[1, 1], [-1, 1]]) @ theta_beta + theta_0.reshape(2, 1)

    phi_r = phi_r.reshape(4, -1)    # 4*n
    phi_l = phi_l.reshape(4, -1)    # 4*n

    #### Tramsform (motor angle from 0 to initial pose) ####
    tramsform_r = []
    tramsform_l = []
    for i in range(4):
        tramsform_r.append( np.hstack((np.linspace(0, phi_r[i, 0], 2000), phi_r[i, 0]*np.ones(0))) )  # finally 4*m
        tramsform_l.append( np.hstack((np.linspace(0, phi_l[i, 0], 2000), phi_l[i, 0]*np.ones(0))) )
    phi_r = np.hstack((tramsform_r, phi_r))
    phi_l = np.hstack((tramsform_l, phi_l))


    # put into the format of motor command
    motor_command = np.empty((phi_r.shape[1], 2*phi_r.shape[0]))
    for i in range(4):
        motor_command[:, 2*i] = phi_r[i, :]
        motor_command[:, 2*i+1] = phi_l[i, :]

    # ## This Issue Have Been Fixed By Yi-Syuan In Webots ##
    # # transfer motor command to be continuous, i.e. [pi-d, -pi+d] -> [pi-d, pi+d]
    # threshold = pi/2
    # last = motor_command[0,:]
    # for angle in motor_command[1:]:
    #     for i in range(8):
    #         while np.abs(angle[i]-last[i]) > threshold: 
    #             angle[i] -= pi*np.sign(angle[i]-last[i]) 
    #     last = angle  

    # write motor commands into xlsx file #

    motor_command = np.hstack(( motor_command, -1*np.ones((motor_command.shape[0], 4)) ))    # add four column of -1 
    df = pd.DataFrame(motor_command)

    # 將 DataFrame 寫入 Excel 檔案
    df.to_csv(file_name + '.csv', index=False, header=False)
    

def inv_kinematics(tiptoe):    # position of G relative to hip
    foot_length = np.linalg.norm(tiptoe)
    theta = np.polyval(G_coef_i, foot_length)    # theta
    beta = np.angle( (tiptoe[0]+1j*tiptoe[1]) / -1j*foot_length )  # beta
    return theta, beta
    
def inv_kinematics_lower(foothold):    # foothold of left lower rim relative to hip
    center = foothold + np.array([0, outer_radius])   # center of lower rim
    theta = np.polyval(O2_r_coef_i, np.linalg.norm(center))    # theta
    beta0_center = [np.polyval(O2_x_coef, theta), np.polyval(O2_y_coef, theta)]
    beta = np.angle( (center[0]+1j*center[1]) / (beta0_center[0]+1j*beta0_center[1]) )  # beta
    contact_map.mapping(theta, beta)
    if contact_map.rim == 3:    # if contact with G rather than lower rim
        theta, beta = inv_kinematics(foothold + np.array([0, r])) 
    return theta, beta
    
def get_foothold(theta, beta, contact_rim=0):
    contact_map.mapping(theta, beta)
    contact_rim = contact_map.rim if contact_map.rim in [2, 3, 4] else 2
    
    if contact_rim in [1, 5]:    # upper rim
        if contact_rim == 1:    # left
            center_beta0 = np.polyval(O1_x_coef, theta) +1j*np.polyval(O1_y_coef, theta)    
        else:  # right
            center_beta0 = -np.polyval(O1_x_coef, theta) +1j*np.polyval(O1_y_coef, theta)    
        center_exp = center_beta0 * np.exp( 1j*beta )
        return np.array([center_exp.real, center_exp.imag - outer_radius])
    elif contact_rim in [2, 4]: # lower rim
        if contact_rim == 2:    # left
            center_beta0 = np.polyval(O2_x_coef, theta) +1j*np.polyval(O2_y_coef, theta)    
        else:  # right
            center_beta0 = -np.polyval(O2_x_coef, theta) +1j*np.polyval(O2_y_coef, theta)    
        center_exp = center_beta0 * np.exp( 1j*beta )
        return np.array([center_exp.real, center_exp.imag - outer_radius])   
    elif contact_rim == 3: # G
        center_beta0 = -1j*np.polyval(G_coef, theta) # G
        center_exp = center_beta0 * np.exp( 1j*beta )
        return np.array([center_exp.real, center_exp.imag - r]) 
    
    print("ERROR IN get_foothold")
    return np.array([0, 0])   

#### Move CoM to the stable position (exceed tip-over axis) ####
def move_CoM_stable(leg_info, swing_leg, CoM, pitch, stability_margin, dS, CoM_bias, theta_list, beta_list, hip_list):
    CoM_offset = np.array([[np.cos(pitch), -np.sin(pitch)],
                            [np.sin(pitch),  np.cos(pitch)]]) @ CoM_bias.reshape(-1, 1)
    CoM_offset = CoM_offset.reshape(-1)
    if leg_info[swing_leg].ID in [0, 1]:
        while CoM[0] + CoM_offset[0] > (leg_info[(swing_leg+1)%4].foothold[0]+leg_info[(swing_leg-1)%4].foothold[0])/2 - stability_margin:
            CoM -= dS * np.array([np.cos(0), np.sin(0)])
            for i in range(4):
                hip = leg_info[i].hip_position(CoM, pitch)
                theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
                leg_info[i].foothold = hip + get_foothold(theta, beta)
                theta_list[i].append(theta)
                beta_list[i].append(beta)
                hip_list[i].append(hip.copy())  
    elif leg_info[swing_leg].ID in [2, 3]:
        while CoM[0] + CoM_offset[0] < (leg_info[(swing_leg+1)%4].foothold[0]+leg_info[(swing_leg-1)%4].foothold[0])/2 + stability_margin:
            CoM += dS * np.array([np.cos(0), np.sin(0)])
            for i in range(4):
                hip = leg_info[i].hip_position(CoM, pitch)
                theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
                leg_info[i].foothold = hip + get_foothold(theta, beta)
                theta_list[i].append(theta)
                beta_list[i].append(beta)
                hip_list[i].append(hip.copy())
    else:
        print('ERROR IN move_CoM_stable')
                
#### Move up CoM ####            
def move_up_CoM(leg_info, CoM, pitch, final_CoM, final_pitch, dS, theta_list, beta_list, hip_list):
    samples = int((final_CoM[1] - CoM[1])//dS)
    delta_CoM = (final_CoM - CoM)/samples
    delta_pitch = (final_pitch - pitch)/samples
    for _ in range(samples):
        pitch += delta_pitch
        CoM += delta_CoM
        for i in range(4):
            hip = leg_info[i].hip_position(CoM, pitch)
            theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
            leg_info[i].foothold = hip + get_foothold(theta, beta)
            theta_list[i].append(theta)
            beta_list[i].append(beta)
            hip_list[i].append(hip.copy())
            
step_h = 0.05   # leg step height on ground
def swing_without_hip_move(leg_info, swing_leg, CoM, pitch, samples, theta_list, beta_list, hip_list):
    hip = leg_info[swing_leg].hip_position(CoM, pitch)
    p_lo = leg_info[swing_leg].foothold + np.array([0, r])   # lift point (G) in world coordinate
    p_td = leg_info[swing_leg].next_foothold + np.array([0, r]) # touch point (G) in world coordinate
    sp = SwingProfile(p_td[0] - p_lo[0], step_h, 0.0, 0.0, 0.0, 0.0, 0.0, p_lo[0], p_lo[1])
    for sample in range(samples):
        swing_phase_ratio = (sample+1)/samples
        curve_point = sp.getFootendPoint(swing_phase_ratio) # G position in world coordinate
        theta, beta = inv_kinematics(curve_point - hip)
        theta_list[swing_leg].append(theta)
        beta_list[swing_leg].append(beta)
        hip_list[swing_leg].append(hip.copy())
        for i in set(range(4)) - {swing_leg}:
            theta_list[i].append(theta_list[i][-1])
            beta_list[i].append(beta_list[i][-1])
            hip_list[i].append(hip_list[i][-1]) 
            
            
# def swing_next_step_ver1(leg_info, swing_leg, CoM, pitch, current_foot_length, current_angle, landing_CoM, landing_pitch, landing_foot_length, landing_angle, samples2, theta_list, beta_list, hip_list):
#     swing_length_arr = parabolic_blends([current_foot_length, 0.1, landing_foot_length], [0, 0.5, 1.0], tp=0.1)
#     swing_angle_arr = parabolic_blends([current_angle, (current_angle+landing_angle)/2, landing_angle, landing_angle], [0, 0.4, 0.8, 1.0], tp=0.1)   
#     delta_pitch = (landing_pitch - pitch)/samples2
#     delta_CoM = (landing_CoM - CoM)/samples2
#     for sample in range(samples2):
#         swing_phase_ratio = (sample+1)/samples2
#         pitch += delta_pitch
#         CoM += delta_CoM
#         foot_length = get_parabolic_point(swing_phase_ratio, swing_length_arr, [0, 0.5, 1.0], tp=0.1)
#         theta = np.polyval(G_coef_i, foot_length)    # theta
#         beta = get_parabolic_point(swing_phase_ratio, swing_angle_arr,[0, 0.4, 0.8, 1.0],  tp=0.1)
#         theta_list[swing_leg].append(theta)
#         beta_list[swing_leg].append(beta)
#         hip = leg_info[swing_leg].hip_position(CoM, pitch)
#         hip_list[swing_leg].append(hip.copy())
#         for i in set(range(4)) - {swing_leg}:
#             hip = leg_info[i].hip_position(CoM, pitch)
#             theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
#             leg_info[i].foothold = hip + get_foothold(theta, beta)
#             theta_list[i].append(theta)
#             beta_list[i].append(beta)
#             hip_list[i].append(hip.copy()) 

# def swing_next_step_ver2(leg_info, swing_leg, CoM, pitch, landing_CoM, landing_pitch, samples, sampling, theta_list, beta_list, hip_list):
#     current_hip = hip_list[swing_leg][-1]
#     current_theta = theta_list[swing_leg][-1]
#     current_beta = beta_list[swing_leg][-1]

#     final_hip = leg_info[swing_leg].hip_position(landing_CoM, landing_pitch)
#     final_theta, final_beta = inv_kinematics_lower(leg_info[swing_leg].next_foothold - final_hip)
    
#     # 0 ~ 0.7 by linear interpolation of foot length and beta
#     mid_ratio = 0.7
#     current_foot_length = np.linalg.norm(leg_info[swing_leg].foothold+np.array([0, r]) - hip_list[swing_leg][-1])
#     mid_hip = current_hip + mid_ratio*(final_hip-current_hip)
#     mid_foot_length = 0.1 + (np.polyval(G_coef, final_theta) - 0.1) *((mid_ratio-0.5)*2)
#     mid_theta = np.polyval(G_coef_i, mid_foot_length)
#     mid_beta = current_beta + (final_beta - 2*np.pi - current_beta) *mid_ratio
#     tp1 = 0.1
#     swing_length_arr = parabolic_blends([current_foot_length, 0.1, mid_foot_length], [0, 0.5, 1.0], tp=tp1, vf=None)
#     swing_angle_arr = parabolic_blends([current_beta, (current_beta+mid_beta)/2, mid_beta], [0, 0.5, 1.0], tp=tp1, vf=None)
    
#     # last point & velocity of linear interpolation
#     foot_length_last1 = get_parabolic_point(1.0, swing_length_arr, [0, 0.5, 1.0], tp=tp1)
#     theta_last1 = np.polyval(G_coef_i, foot_length_last1)    # theta
#     beta_last1 = get_parabolic_point(1.0, swing_angle_arr, [0, 0.5, 1.0],  tp=tp1)  
#     hip_last1 = mid_hip
#     contact_map.calculate(theta_last1, beta_last1)
#     G_last1 = hip_last1 + np.array([linkleg.G.real, linkleg.G.imag])
#     O2_last1 = hip_last1 + np.array([linkleg.O2.real, linkleg.O2.imag])
    
#     foot_length_last2 = get_parabolic_point(1.0-(1/samples/mid_ratio), swing_length_arr, [0, 0.5, 1.0], tp=tp1)
#     theta_last2 = np.polyval(G_coef_i, foot_length_last2)    # theta
#     beta_last2 = get_parabolic_point(1.0-(1/samples/mid_ratio), swing_angle_arr, [0, 0.5, 1.0],  tp=tp1) 
#     hip_last2 = current_hip + (mid_ratio-1/samples)*(final_hip-current_hip)
#     contact_map.calculate(theta_last2, beta_last2)
#     G_last2 = hip_last2 + np.array([linkleg.G.real, linkleg.G.imag])
#     O2_last2 = hip_last2 + np.array([linkleg.O2.real, linkleg.O2.imag])
     
#     # 0.7 ~ 1.0 by parabolic blend
#     # contact_map.calculate(mid_theta, mid_beta)
#     # mid_G = mid_hip + np.array([linkleg.G.real, linkleg.G.imag])
#     # mid_O2 = mid_hip + np.array([linkleg.O2.real, linkleg.O2.imag])
#     contact_map.calculate(final_theta, final_beta)
#     if contact_map.rim == 3:    # if contact with G rather than lower rim
#         contact_with_G = True
#         final_G = final_hip + np.array([linkleg.G.real, linkleg.G.imag])
#         tp = 0.3
#         x_t = [0, 0.5, 1.0]
#         y_t = [0, 0.5, 1.0]
#         v_x = (G_last1[0] - G_last2[0]) * sampling * ((1-mid_ratio)*samples/sampling)
#         v_y = (G_last1[1] - G_last2[1]) * sampling * ((1-mid_ratio)*samples/sampling)
#         # x_cof = parabolic_blends([mid_G[0], final_G[0], final_G[0]], x_t, tp=tp)
#         # y_cof = parabolic_blends([mid_G[1], (mid_G[1]+final_G[1])/2, final_G[1]], y_t, tp=tp)    
#         x_cof = parabolic_blends([G_last1[0], final_G[0], final_G[0]], x_t, tp=tp, vi=v_x)
#         y_cof = parabolic_blends([G_last1[1], (G_last1[1]+final_G[1])/2, final_G[1]], y_t, tp=tp, vi=v_y)    
#         print(v_x, v_y, 'G')
#     else:   # contact with lower rim
#         contact_with_G = False
#         final_O2 = final_hip + np.array([linkleg.O2.real, linkleg.O2.imag])
#         tp = 0.3
#         x_t = [0, 0.5, 1.0]
#         y_t = [0, 0.5, 1.0]
#         v_x = (O2_last1[0] - O2_last2[0]) * sampling * ((1-mid_ratio)*samples/sampling)
#         v_y = (O2_last1[1] - O2_last2[1]) * sampling * ((1-mid_ratio)*samples/sampling)
#         # x_cof = parabolic_blends([mid_O2[0], final_O2[0], final_O2[0]], x_t, tp=tp)
#         # y_cof = parabolic_blends([mid_O2[1], (mid_O2[1]+final_O2[1])/2, final_O2[1]], y_t, tp=tp)   
#         x_cof = parabolic_blends([O2_last1[0], final_O2[0], final_O2[0]], x_t, tp=tp, vi=v_x)
#         y_cof = parabolic_blends([O2_last1[1], (O2_last1[1]+final_O2[1])/2, final_O2[1]], y_t, tp=tp, vi=v_y)     
#         print(v_x, v_y, 'Lower')
    
#     p_size = 0.2    # point size of each hip position
#     fig_size = 10
#     fig, ax = plt.subplots()
#     # fig, ax = plt.subplots(figsize = (fig_size,fig_size))
#     plt.grid(True)
#     ax.set_aspect('equal')  # 座標比例相同
#     d = np.linspace(0, 1, 10000)
#     x_points = [get_parabolic_point(i, x_cof, x_t, tp=tp) for i in d]
#     y_points = [get_parabolic_point(i, y_cof, y_t, tp=tp) for i in d]
#     ax.plot(x_points, y_points)
#     kk = 0.1
#     ax.plot(*(get_parabolic_point(kk, x_cof, x_t, tp=tp), get_parabolic_point(kk, y_cof, y_t, tp=tp)), '*r')
#     plt.show()
    
#     delta_pitch = (landing_pitch - pitch)/samples
#     delta_CoM = (landing_CoM - CoM)/samples
#     for sample in range(samples):
#         swing_phase_ratio = (sample+1)/samples
#         pitch += delta_pitch
#         CoM += delta_CoM
#         hip = leg_info[swing_leg].hip_position(CoM, pitch)
#         if swing_phase_ratio <= mid_ratio:
#             swing_phase_ratio /= mid_ratio
#             foot_length = get_parabolic_point(swing_phase_ratio, swing_length_arr, [0, 0.5, 1.0], tp=tp1)
#             theta = np.polyval(G_coef_i, foot_length)    # theta
#             beta = get_parabolic_point(swing_phase_ratio, swing_angle_arr, [0, 0.5, 1.0],  tp=tp1)  
#         else:   # 0.7~1.0
#             swing_phase_ratio = (swing_phase_ratio - mid_ratio) / (1.0 - mid_ratio)
#             x_p = get_parabolic_point(swing_phase_ratio, x_cof, x_t, tp=tp) # G or O2 x
#             y_p = get_parabolic_point(swing_phase_ratio, y_cof, y_t, tp=tp) # G or O2 y
#             if contact_with_G:
#                 theta = np.polyval(G_coef_i, np.linalg.norm(np.array([x_p, y_p]) - hip))    # theta
#                 beta = np.angle( (x_p-hip[0]+1j*(y_p-hip[1])) / -1j )
#             else:
#                 theta = np.polyval(O2_r_coef_i, np.linalg.norm(np.array([x_p, y_p]) - hip))    # theta
#                 x_beta0 = np.polyval(O2_x_coef, theta) # O2 x when (theta, 0)
#                 y_beta0 = np.polyval(O2_y_coef, theta) # O2 y when (theta, 0)
#                 beta = np.angle( (x_p-hip[0]+1j*(y_p-hip[1])) / (x_beta0+1j*y_beta0) )
#         theta_list[swing_leg].append(theta)
#         beta_list[swing_leg].append(beta)
#         hip_list[swing_leg].append(hip.copy())
#         for i in set(range(4)) - {swing_leg}:
#             hip = leg_info[i].hip_position(CoM, pitch)
#             theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
#             leg_info[i].foothold = hip + get_foothold(theta, beta)
#             theta_list[i].append(theta)
#             beta_list[i].append(beta)
#             hip_list[i].append(hip.copy()) 
            
            
            
def swing_next_step(leg_info, swing_leg, CoM, pitch, landing_CoM, landing_pitch, samples, sampling, theta_list, beta_list, hip_list):
    current_hip = hip_list[swing_leg][-1]
    current_theta = theta_list[swing_leg][-1]
    current_beta = beta_list[swing_leg][-1]

    final_hip = leg_info[swing_leg].hip_position(landing_CoM, landing_pitch)
    final_theta, final_beta = inv_kinematics_lower(leg_info[swing_leg].next_foothold - final_hip)
    
    # 0 ~ 0.2 by G vertical up
    first_ratio = 0.1
    current_G = leg_info[swing_leg].foothold + np.array([0, r])
    first_G = current_G + np.array([0, 0.01])
    first_hip = current_hip + first_ratio*(final_hip-current_hip)
    tp1 = 0.3
    x_t1 = [0, 0.5, 1.0]
    y_t1 = [0, 0.5, 1.0]
    G_x_cof = parabolic_blends([current_G[0], first_G[0], first_G[0]], x_t1, tp=tp1, vf=None)
    G_y_cof = parabolic_blends([current_G[1], (current_G[1]+first_G[1])/2, first_G[1]], y_t1, tp=tp1, vf=None)    
    # last point & velocity of linear interpolation
    G_x_last1 = get_parabolic_point(1.0, G_x_cof, [0, 0.5, 1.0], tp=tp1)
    G_y_last1 = get_parabolic_point(1.0, G_y_cof, [0, 0.5, 1.0], tp=tp1)
    hip_last1 = first_hip
    theta_last1 = np.polyval(G_coef_i, np.linalg.norm(np.array([G_x_last1, G_y_last1])-hip_last1))
    beta_last1 = np.angle( (G_x_last1-hip_last1[0]+1j*(G_y_last1-hip_last1[1])) / -1j )
    G_x_last2 = get_parabolic_point(1.0-(1/samples/first_ratio), G_x_cof, [0, 0.5, 1.0], tp=tp1)
    G_y_last2 = get_parabolic_point(1.0-(1/samples/first_ratio), G_y_cof, [0, 0.5, 1.0], tp=tp1)
    hip_last2 = current_hip + (first_ratio-1/samples)*(final_hip-current_hip)
    theta_last2 = np.polyval(G_coef_i, np.linalg.norm(np.array([G_x_last2, G_y_last2])-hip_last2))
    beta_last2 = np.angle( (G_x_last2-hip_last2[0]+1j*(G_y_last2-hip_last2[1])) / -1j )

    
    # 0.2 ~ 0.7 by theta and beta
    mid_ratio = 0.8
    v_theta = (theta_last1 - theta_last2) * sampling * ((mid_ratio-first_ratio)*samples/sampling)
    v_beta = (beta_last1 - beta_last2) * sampling * ((mid_ratio-first_ratio)*samples/sampling)
    mid_hip = current_hip + mid_ratio*(final_hip-current_hip)
    mid_theta = np.deg2rad(17) + (final_theta - np.deg2rad(17)) * ((mid_ratio-0.5)*2)
    mid_beta = current_beta + (final_beta - 2*np.pi - current_beta) *mid_ratio
    tp2 = 0.1
    theta_cof = parabolic_blends([theta_last1, np.deg2rad(17), mid_theta], [0, 0.5, 1.0], tp=tp2, vi=v_theta, vf=None)
    beta_cof = parabolic_blends([beta_last1, (current_beta+mid_beta)/2, mid_beta], [0, 0.5, 1.0], tp=tp2, vi=v_beta, vf=None)
    # last point & velocity of linear interpolation
    theta_last1 = get_parabolic_point(1.0, theta_cof, [0, 0.5, 1.0], tp=tp2)
    beta_last1 = get_parabolic_point(1.0, beta_cof, [0, 0.5, 1.0],  tp=tp2)  
    hip_last1 = mid_hip
    contact_map.calculate(theta_last1, beta_last1)
    G_last1 = hip_last1 + np.array([linkleg.G.real, linkleg.G.imag])
    O2_last1 = hip_last1 + np.array([linkleg.O2.real, linkleg.O2.imag])
    theta_last2 = get_parabolic_point(1.0-(1/samples/(mid_ratio-first_ratio)), theta_cof, [0, 0.5, 1.0], tp=tp2)
    beta_last2 = get_parabolic_point(1.0-(1/samples/(mid_ratio-first_ratio)), beta_cof, [0, 0.5, 1.0],  tp=tp2) 
    hip_last2 = current_hip + (mid_ratio-1/samples)*(final_hip-current_hip)
    contact_map.calculate(theta_last2, beta_last2)
    G_last2 = hip_last2 + np.array([linkleg.G.real, linkleg.G.imag])
    O2_last2 = hip_last2 + np.array([linkleg.O2.real, linkleg.O2.imag])
     
    # 0.7 ~ 1.0 by parabolic blend
    contact_map.calculate(final_theta, final_beta)
    tp3 = 0.3
    if contact_map.rim == 3:    # if contact with G rather than lower rim
        contact_with_G = True
        final_G = final_hip + np.array([linkleg.G.real, linkleg.G.imag])
        x_t = [0, 0.5, 1.0]
        y_t = [0, 0.5, 1.0]
        v_x = (G_last1[0] - G_last2[0]) * sampling * ((1-mid_ratio)*samples/sampling)
        v_y = (G_last1[1] - G_last2[1]) * sampling * ((1-mid_ratio)*samples/sampling)   
        x_cof = parabolic_blends([G_last1[0], final_G[0], final_G[0]], x_t, tp=tp3, vi=v_x)
        y_cof = parabolic_blends([G_last1[1], (G_last1[1]+final_G[1])/2, final_G[1]], y_t, tp=tp3, vi=v_y)    
        print(v_x, v_y, 'G')
    else:   # contact with lower rim
        contact_with_G = False
        final_O2 = final_hip + np.array([linkleg.O2.real, linkleg.O2.imag])
        x_t = [0, 0.5, 1.0]
        y_t = [0, 0.5, 1.0]
        v_x = (O2_last1[0] - O2_last2[0]) * sampling * ((1-mid_ratio)*samples/sampling)
        v_y = (O2_last1[1] - O2_last2[1]) * sampling * ((1-mid_ratio)*samples/sampling) 
        x_cof = parabolic_blends([O2_last1[0], final_O2[0], final_O2[0]], x_t, tp=tp3, vi=v_x)
        y_cof = parabolic_blends([O2_last1[1], (3*O2_last1[1]+final_O2[1])/4, final_O2[1]], y_t, tp=tp3, vi=v_y)     
        print(v_x, v_y, 'Lower')
    
    p_size = 0.2    # point size of each hip position
    fig_size = 10
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize = (fig_size,fig_size))
    plt.grid(True)
    ax.set_aspect('equal')  # 座標比例相同
    d = np.linspace(0, 1, 10000)
    x_points = [get_parabolic_point(i, x_cof, x_t, tp=tp3) for i in d]
    y_points = [get_parabolic_point(i, y_cof, y_t, tp=tp3) for i in d]
    ax.plot(x_points, y_points)
    kk = 0.1
    ax.plot(*(get_parabolic_point(kk, x_cof, x_t, tp=tp3), get_parabolic_point(kk, y_cof, y_t, tp=tp3)), '*r')
    plt.show()
    
    delta_pitch = (landing_pitch - pitch)/samples
    delta_CoM = (landing_CoM - CoM)/samples
    for sample in range(samples):
        swing_phase_ratio = (sample+1)/samples
        pitch += delta_pitch
        CoM += delta_CoM
        hip = leg_info[swing_leg].hip_position(CoM, pitch)
        if swing_phase_ratio < first_ratio:
            swing_phase_ratio = swing_phase_ratio / first_ratio
            x_p = get_parabolic_point(swing_phase_ratio, G_x_cof, x_t1, tp=tp1) # G or O2 x
            y_p = get_parabolic_point(swing_phase_ratio, G_y_cof, y_t1, tp=tp1) # G or O2 y
            theta = np.polyval(G_coef_i, np.linalg.norm(np.array([x_p, y_p]) - hip))    # theta
            beta = np.angle( (x_p-hip[0]+1j*(y_p-hip[1])) / -1j )
        elif swing_phase_ratio <= mid_ratio: # 0.2~0.7
            swing_phase_ratio = (swing_phase_ratio - first_ratio) / (mid_ratio - first_ratio)
            theta = get_parabolic_point(swing_phase_ratio, theta_cof, [0, 0.5, 1.0], tp=tp2)
            beta = get_parabolic_point(swing_phase_ratio, beta_cof, [0, 0.5, 1.0],  tp=tp2)  
        else:   # 0.7~1.0
            swing_phase_ratio = (swing_phase_ratio - mid_ratio) / (1.0 - mid_ratio)
            x_p = get_parabolic_point(swing_phase_ratio, x_cof, x_t, tp=tp3) # G or O2 x
            y_p = get_parabolic_point(swing_phase_ratio, y_cof, y_t, tp=tp3) # G or O2 y
            if contact_with_G:
                theta = np.polyval(G_coef_i, np.linalg.norm(np.array([x_p, y_p]) - hip))    # theta
                beta = np.angle( (x_p-hip[0]+1j*(y_p-hip[1])) / -1j )
            else:
                theta = np.polyval(O2_r_coef_i, np.linalg.norm(np.array([x_p, y_p]) - hip))    # theta
                x_beta0 = np.polyval(O2_x_coef, theta) # O2 x when (theta, 0)
                y_beta0 = np.polyval(O2_y_coef, theta) # O2 y when (theta, 0)
                beta = np.angle( (x_p-hip[0]+1j*(y_p-hip[1])) / (x_beta0+1j*y_beta0) )
        theta_list[swing_leg].append(theta)
        beta_list[swing_leg].append(beta)
        hip_list[swing_leg].append(hip.copy())
        for i in set(range(4)) - {swing_leg}:
            hip = leg_info[i].hip_position(CoM, pitch)
            theta, beta = inv_move_kinematics.inv_move_kinematics([theta_list[i][-1], beta_list[i][-1]], hip_list[i][-1], hip)
            leg_info[i].foothold = hip + get_foothold(theta, beta)
            theta_list[i].append(theta)
            beta_list[i].append(beta)
            hip_list[i].append(hip.copy()) 