import numpy as np
import Contact_Map

#### Parameters ####
# contact_points: n contact points relative to hip. (x, y): complex number (real, imag)
# theta_list: n theta correspond to n contact point.
# beta_list: n beta correspond to n contact point.
# touch_point: leg contact the ground at (x, y) in world coordinate.
# W: stair width(m)
# H: stair height(m)
def collision_check(contact_points, theta_list, beta_list, touch_point=np.array([[.0], [.0]]), W=0.27, H=0.17, stair_edge = np.array([0, 0])):    
    contact_map = Contact_Map.ContactMap()
    linkleg = contact_map.leg.linkleg
    r = linkleg.r
    outer_radius = linkleg.R + r
    radius = linkleg.R
    
    touch_point = touch_point.reshape(2, -1)
    stair_edge = [np.array(stair_edge), np.array(stair_edge) + [W, H]]
    check_distance = [r, outer_radius, outer_radius, outer_radius, outer_radius]    # distance from center (G, O1, O2) to wheel rim

    x = contact_points.real
    y = contact_points.imag
    hip_positions = touch_point - np.array([x, y])
    n_amount = hip_positions.shape[1]

    contact_map.mapping(theta_list, beta_list)

    collision = np.zeros(n_amount, dtype=bool)    # if this point would cause collision between wheel and stair.
    for idx, hip in enumerate(np.transpose(hip_positions, (1, 0))): # for each hip position
        for c, leg_part in enumerate([linkleg.G[idx], linkleg.O1[idx], linkleg.O2[idx], linkleg.O1_c[idx], linkleg.O2_c[idx]]):    # check each part of leg.
            x = leg_part.real
            y = leg_part.imag
            center = hip + [x, y]
            for edge in stair_edge:
                if hip[0] > edge[0] and hip[1] < edge[1]:
                    collision[idx] = True
                    break
                
                if center[1] > edge[1]:   # y > H
                    if center[0] > edge[0]:    # x > W 
                    # Quadrant 1
                        distance = [0, edge[1] - center[1]] # distance from center to stair horizon plane
                    else:   # x <= W 
                    # Quadrant 2                
                        distance = [edge[0], edge[1]] - center # distance from center to edge of stair
                else:   # y <= H
                    if center[0] < edge[0]:    # x < W 
                    # Quadrant 3
                        distance = [edge[0] - center[0], 0] # distance from center to stair vertical plane
                    else:   # x >= W 
                    # Quadrant 4                
                        distance = [0.0001, -0.0001] # consideredd very close to the edge of stair to check the range to collide.
                        # collision[idx] = True
                        # continue
                
                # Check if rim is in the range to collide.
                in_range = True
                distance = distance[0] + 1j*distance[1]
                OH = linkleg.H[idx].real - x + 1j*(linkleg.H[idx].imag - y)
                OH_c = linkleg.H_c[idx].real - x + 1j*(linkleg.H_c[idx].imag - y)
                OF = linkleg.F[idx].real - x + 1j*(linkleg.F[idx].imag - y)
                OF_c = linkleg.F_c[idx].real - x + 1j*(linkleg.F_c[idx].imag - y)
                OG = linkleg.G[idx].real - x + 1j*(linkleg.G[idx].imag - y)
                if c == 1:   # left upper rim
                    angle1 = np.angle(OF/distance)
                    angle2 = np.angle(distance/OH)
                    if (angle1 < 0):
                        if (np.linalg.norm(OF) + r) * np.cos(angle1) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue
                    elif (angle2 < 0): # when longest direction is between hip and H
                        if (np.linalg.norm(OH) + r) * np.cos(angle2) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue
                elif c == 2:   # left lower rim
                    angle1 = np.angle(OG/distance)
                    angle2 = np.angle(distance/OF)
                    if (angle1 < 0):
                        in_range = False  
                    elif  (angle2 < 0):
                        if (np.linalg.norm(OF) + r) * np.cos(angle2) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue    
                elif c == 3:   # right upper rim        
                    angle1 = np.angle(distance/OF_c)
                    angle2 = np.angle(OH_c/distance)
                    if (angle1 < 0):
                        if (np.linalg.norm(OF_c) + r) * np.cos(angle1) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue
                    elif (angle2 < 0): # when longest direction is between hip and H
                        if (np.linalg.norm(OH_c) + r) * np.cos(angle2) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue
                elif c == 4:   # right lower rim
                    angle1 = np.angle(distance/OG)
                    angle2 = np.angle(OF_c/distance)
                    if (angle1 < 0):
                        in_range = False     
                    elif  (angle2 < 0):
                        if (np.linalg.norm(OF_c) + r) * np.cos(angle2) >= np.linalg.norm(distance):
                            collision[idx] = True 
                        continue    
                        
                if in_range:
                    if np.linalg.norm(distance) < check_distance[c]*0.999999:
                        collision[idx] = True 
                        
        if idx % (n_amount//10) == 0:
            print('.', end="")
    return collision