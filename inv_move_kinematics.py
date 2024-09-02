import numpy as np
import Contact_Map
from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import fsolve
from FittingCoefficient import *

contact_map = Contact_Map.ContactMap()
linkleg = contact_map.leg.linkleg
r = linkleg.r
outer_radius = linkleg.R + r
radius = linkleg.R

def objective(d_q, last_q, last_hip, hip): # [d_teata, d_beta], [last_teata, last_beta], [last_x, last_y], [x, y]
    new_q = last_q + d_q
    foot_length = np.polyval(G_coef, [last_q[0], new_q[0]]) # [last, new]
    # contact_map.mapping(*(last_q))
    # contact_map.calculate(*(last_q))
    last_G_exp = -1j*foot_length[0] *np.exp( 1j*(last_q[1]) ) # in polor coordinate (hip:[0,0])
    last_G = np.array([last_G_exp.real, last_G_exp.imag]) # in leg coordinate (hip:[0,0])
    # last_G = np.array([linkleg.G.real, linkleg.G.imag]) # in leg coordinate (hip:[0,0])
    last_G_world = last_hip + last_G    # in world coordinate
    roll_forward = (-d_q[1]*r)  # roll forward distance caused by rotation of contact point
    G_world = last_G_world + np.array([roll_forward, 0])   # G = last_G + roll forward
    
    # contact_map.mapping(*(last_q+d_q))
    # contact_map.calculate(*(last_q+d_q))
    G_exp = -1j*foot_length[1] *np.exp( 1j*(new_q[1]) ) # in polor coordinate (hip:[0,0])
    G = np.array([G_exp.real, G_exp.imag]) # in leg coordinate (hip:[0,0])
    # G = np.array([linkleg.G.real, linkleg.G.imag]) # in leg coordinate (hip:[0,0])
    guessed_hip = G_world - G
    # return np.linalg.norm(guessed_hip-hip)
    return guessed_hip-hip
    

def objective2(d_q, last_q, last_hip, hip): # [d_teata, d_beta], [last_teata, last_beta], [last_x, last_y], [x, y]
    new_q = last_q + d_q
    last_G_exp, G_exp = (-1j *np.polyval(G_coef, [last_q[0], new_q[0]])) *np.exp( [1j*(last_q[1]), 1j*(new_q[1])] ) # [last, new]
    center_x = np.polyval(O2_x_coef, [last_q[0], new_q[0]]) # [last, new]
    center_y = np.polyval(O2_y_coef, [last_q[0], new_q[0]]) # [last, new]
    last_center_exp = (center_x[0] + 1j *center_y[0]) *np.exp( 1j*(last_q[1]) )
    center_exp = (center_x[1] + 1j *center_y[1]) *np.exp( 1j*(new_q[1]) )
    
    last_alpha = np.deg2rad(50) - np.angle( (last_G_exp - last_center_exp) / -1j)
    alpha = np.deg2rad(50) - np.angle( (G_exp - center_exp) / -1j)

    roll_forward = (alpha - last_alpha)*outer_radius  # roll forward distance caused by rotation of contact point

    last_center_world = last_hip + np.array([last_center_exp.real, last_center_exp.imag])
    center_world = last_center_world + np.array([roll_forward, 0])
    guessed_hip = center_world - np.array([center_exp.real, center_exp.imag])
    # return np.linalg.norm(guessed_hip-hip)
    return guessed_hip-hip
    

def objective_r_lower(d_q, last_q, last_hip, hip): # [d_teata, d_beta], [last_teata, last_beta], [last_x, last_y], [x, y]
    new_q = last_q + d_q
    last_G_exp, G_exp = (-1j *np.polyval(G_coef, [last_q[0], new_q[0]])) *np.exp( [1j*(last_q[1]), 1j*(new_q[1])] ) # [last, new]
    center_x = -np.polyval(O2_x_coef, [last_q[0], new_q[0]]) # [last, new]
    center_y = np.polyval(O2_y_coef, [last_q[0], new_q[0]]) # [last, new]
    last_center_exp = (center_x[0] + 1j *center_y[0]) *np.exp( 1j*(last_q[1]) )
    center_exp = (center_x[1] + 1j *center_y[1]) *np.exp( 1j*(new_q[1]) )
    
    last_alpha = np.angle( -1j / (last_G_exp - last_center_exp))
    alpha = np.angle( -1j / (G_exp - center_exp))

    roll_forward = (alpha - last_alpha)*outer_radius  # roll forward distance caused by rotation of contact point

    last_center_world = last_hip + np.array([last_center_exp.real, last_center_exp.imag])
    center_world = last_center_world + np.array([roll_forward, 0])
    guessed_hip = center_world - np.array([center_exp.real, center_exp.imag])
    # return np.linalg.norm(guessed_hip-hip)
    return guessed_hip-hip

def inv_move_kinematics(last_q, last_hip, hip, contact_rim=0):  # only for G contacting 
    # if contact_rim == 0 :    # first calculate lowest point as contact point 
    contact_map.mapping(*last_q)
    contact_rim = contact_map.rim if contact_map.rim in [2, 3, 4] else 2
        
    if contact_rim == 2:
        select_objective = objective2       
    elif contact_rim == 3:
        select_objective = objective
    elif contact_rim == 4:
        select_objective = objective_r_lower
    else:
        print("ERROR IN inv_move_kinematics")
    # result = minimize(lambda d_q: select_objective(d_q, last_q, last_hip, hip), np.array([0, 0]), method='BFGS', tol=1e-4, options={'maxiter': 10})
    # d_theta, d_beta = result.x
    # if result.fun > 0.0001: # if error too large
    #         print(result.x, result.fun)
    result = fsolve(lambda d_q: select_objective(d_q, last_q, last_hip, hip), np.array([0, 0]))     
    d_theta, d_beta = result
    return last_q[0]+d_theta, last_q[1]+d_beta
        
    
def inv_move_kinematics_lower(last_q, last_hip, hip):  # only for lower rim contacting 
    result = minimize(lambda d_q: objective2(d_q, last_q, last_hip, hip), np.array([0, 0]), method='BFGS', tol=1e-4, options={'maxiter': 10})
    d_theta, d_beta = result.x
    if result.fun > 0.0001: # if error too large
        print(result.x, result.fun)
    return last_q[0]+d_theta, last_q[1]+d_beta