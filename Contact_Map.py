import numpy as np
import LegModel
  
def arc_min(p1, p2, O, radius, amount): # alpha: arc starting from most clockwise (most left) of the rim
    lowest_points = np.array(.0) if amount == 0 else np.zeros(amount)
    alpha = np.array(.0) if amount == 0 else np.zeros(amount)
    
    in_range = ((p2 - O).real > 0) & ((p1 - O).real < 0)    # lowest point is between 2 endpoints
    lowest_points[in_range] = O[in_range].imag - radius
    alpha[in_range] = np.angle( -1j/(p1[in_range] - O[in_range]) )
    
    p1_out = p1[~in_range]  # lowest point is one of 2 endpoints
    p2_out = p2[~in_range]  # lowest point is one of 2 endpoints
    O_out = O[~in_range]    # lowest point is one of 2 endpoints
    smaller = np.where(p1_out.imag < p2_out.imag, p1_out, p2_out)   # smaller y value of 2 endpoints
    lowest_points[~in_range] = smaller.imag
    alpha[~in_range] = np.angle( (smaller - O_out)/(p1_out - O_out) )
    return lowest_points, alpha
 
def arc_min_G(p1, p2, O, radius, amount):   # special definition: alpha=0 when beta=0 (point directly below G)
    lowest_points = np.array(.0) if amount == 0 else np.zeros(amount)
    alpha = np.array(.0) if amount == 0 else np.zeros(amount)
    direction_G = p1+p2
    bias_alpha = np.angle( direction_G/(p1 - O) )
    
    in_range = ((p2 - O).real > 0) & ((p1 - O).real < 0)    # lowest point is between 2 endpoints
    lowest_points[in_range] = O[in_range].imag - radius
    alpha[in_range] = np.angle( -1j/(p1[in_range] - O[in_range]) )
    
    p1_out = p1[~in_range]  # lowest point is one of 2 endpoints
    p2_out = p2[~in_range]  # lowest point is one of 2 endpoints
    O_out = O[~in_range]    # lowest point is one of 2 endpoints
    smaller = np.where(p1_out.imag < p2_out.imag, p1_out, p2_out)   # smaller y value of 2 endpoints
    lowest_points[~in_range] = smaller.imag
    alpha[~in_range] = np.angle( (smaller - O_out)/(p1_out - O_out) )
    return lowest_points, alpha - bias_alpha
             
class ContactMap:
    def __init__(self, offset = np.array([0, 0, 0])) -> None:
        self.leg = LegModel.Leg(offset)
        self.theta = np.deg2rad(17.0)
        self.beta = 0
        self.rim = 4 # 1 -> 2 -> 3 -> 4 -> 5 -> 6: 
                    # O1 -> O2 -> G -> O2'-> O1'-> None
        self.r = np.array([self.theta, self.beta])
        self.max_theta = self.leg.linkleg.max_theta
        self.min_theta = self.leg.linkleg.min_theta
        
    def update(self, theta_d, beta_d, omega_d, dt):
        self.theta += theta_d * dt
        self.beta += ((beta_d + omega_d) * dt + 2 * np.pi)
        limit_u = self.theta > self.max_theta   # theta exceeding upper bound set to upper bound
        self.theta[limit_u] = self.max_theta
        limit_l = self.theta < self.min_theta   # theta below lower bound set to lower bound
        self.theta[limit_l] = self.min_theta
        self.beta = np.fmod(self.beta, 2 * np.pi)
        self.r = np.array([self.theta, self.beta])
        self.rim_type()
        return self.r
    
    def mapping(self, theta, beta):
        self.theta = np.array(theta).astype(float) 
        limit_u = self.theta > self.max_theta   # theta exceeding upper bound set to upper bound
        self.theta[limit_u] = self.max_theta
        limit_l = self.theta < self.min_theta   # theta below lower bound set to lower bound
        self.theta[limit_l] = self.min_theta

        self.beta = beta + 2 * np.pi
        self.beta = np.fmod(self.beta, 2 * np.pi)
        self.r = np.array([self.theta, self.beta])
        self.rim_type()
        return
    
    def lookup(self):
        return self.rim, self.alpha
    
    def rim_type(self):
        self.leg.calculate(self.theta, 0, 0, self.beta, 0, 0)
        real_R = self.leg.linkleg.R + self.leg.linkleg.r
        self.amount = 0 if self.theta.ndim == 0 else self.theta.shape[0]  # amount of theta given in an array, 0: only 1 scalar.
        O1 = self.leg.linkleg.O1
        O2 = self.leg.linkleg.O2
        O1_c = self.leg.linkleg.O1_c
        O2_c = self.leg.linkleg.O2_c
        G = (self.leg.linkleg.G - O2) / self.leg.linkleg.R * real_R + O2
        G_c = (self.leg.linkleg.G - O2_c) / self.leg.linkleg.R * real_R + O2_c
        H = (self.leg.linkleg.H - O1) / self.leg.linkleg.R * real_R + O1
        H_c = (self.leg.linkleg.H_c - O1_c) / self.leg.linkleg.R * real_R + O1_c
        F = (self.leg.linkleg.F - O1) / self.leg.linkleg.R * real_R + O1
        F_c = (self.leg.linkleg.F_c - O1_c) / self.leg.linkleg.R * real_R + O1_c        
        # 8 cases
        zeros = (0, 0) if self.amount == 0 else (np.zeros(self.amount), np.zeros(self.amount))
        case_list = [
            arc_min(H, F, O1, real_R, self.amount),
            arc_min(F, G, O2, real_R, self.amount),
            arc_min_G(G, G_c, self.leg.linkleg.G, self.leg.linkleg.r, self.amount),
            arc_min(G_c, F_c, O2_c, real_R, self.amount),
            arc_min(F_c, H_c, O1_c, real_R, self.amount),
            zeros
        ]
        case_list = np.array(case_list)

        self.rim = np.argmin(case_list[:, 0], axis=0) + 1   
        self.alpha = case_list[self.rim-1, 1] if self.amount == 0 else case_list[self.rim-1, 1, np.arange(self.amount)]
        self.contact_height = case_list[self.rim-1, 0] if self.amount == 0 else case_list[self.rim-1, 0, np.arange(self.amount)]    # contact point y relative to hip represented in world coordinate.


    def calculate(self, theta, beta):
        self.theta = np.array(theta).astype(float) 
        limit_u = self.theta > self.max_theta   # theta exceeding upper bound set to upper bound
        self.theta[limit_u] = self.max_theta
        limit_l = self.theta < self.min_theta   # theta below lower bound set to lower bound
        self.theta[limit_l] = self.min_theta

        self.beta = beta + 2 * np.pi
        self.beta = np.fmod(self.beta, 2 * np.pi)
        self.leg.calculate(self.theta, 0, 0, self.beta, 0, 0)
        return
#Contact Map .py