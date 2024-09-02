import numpy as np
from LegModel import *
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from Contact_Map import *

class SingleLeg_Animation():
    def __init__(self, O=np.array([0, 0]), offset=np.array([0, 0, 0])):
        self.setting()
        self.ani_leg = self.Ani_Leg(offset, O, self.mark_size, self.line_width)   # all Shapes of leg
        self.contact_map = ContactMap()
        
    class Ani_Leg(Leg):
        def __init__(self, offset, O, mark_size, line_width):
            super().__init__(offset)
            self.O = np.array(O) # origin of leg in world coordinate
            self.mark_size = mark_size
            self.line_width = line_width
            self.R = self.linkleg.R
            self.r = self.linkleg.r
                  
        class rim:
            def __init__(self, arc, arc_out, start):
                self.arc = [arc, arc_out]   # inner & outer arcs
                self.start = start          # start angle
                
        ## Get Shape Of Leg ##  
        def get_shape(self, theta=np.deg2rad(17.0), beta=0):
            self.calculate(theta, 0, 0, beta, 0, 0)
            # four rims (inner arc, outer arc, start angle)
            self.upper_rim_r = self.rim( *self.get_arc(self.linkleg.F_c, self.linkleg.H_c, self.linkleg.O1_c, 'black', self.r))
            self.upper_rim_l = self.rim( *self.get_arc(self.linkleg.H, self.linkleg.F, self.linkleg.O1, 'black', self.r))
            self.lower_rim_r = self.rim( *self.get_arc(self.linkleg.G, self.linkleg.F_c, self.linkleg.O2_c, 'black', self.r))
            self.lower_rim_l = self.rim( *self.get_arc(self.linkleg.F, self.linkleg.G, self.linkleg.O2, 'black', self.r))
            # five joints on the rims   (center, radius)
            self.upper_joint_r = self.get_circle(self.linkleg.H_c, self.r) 
            self.upper_joint_l = self.get_circle(self.linkleg.H, self.r) 
            self.lower_joint_r = self.get_circle(self.linkleg.F_c, self.r) 
            self.lower_joint_l = self.get_circle(self.linkleg.F, self.r) 
            self.G_joint       = self.get_circle(self.linkleg.G, self.r)
            # six bars  (point1, point2)
            self.OB_bar_r = self.get_line(0, self.linkleg.B_c) 
            self.OB_bar_l = self.get_line(0, self.linkleg.B) 
            self.AE_bar_r = self.get_line(self.linkleg.A_c, self.linkleg.E)
            self.AE_bar_l = self.get_line(self.linkleg.A, self.linkleg.E)
            self.CD_bar_r = self.get_line(self.linkleg.C_c, self.linkleg.D_c)
            self.CD_bar_l = self.get_line(self.linkleg.C, self.linkleg.D) 
            
        def get_arc(self, p1, p2, o, color='black', offset=0.01):
            start = np.angle(p1-o, deg=True)
            end = np.angle(p2-o, deg=True)
            radius = np.abs(p1-o)
            arc = Arc([o.real, o.imag], 2*(radius-offset), 2*(radius-offset), angle=0.0, theta1=start, theta2=end, color=color, linewidth=self.line_width)
            arc_out = Arc([o.real, o.imag], 2*(radius+offset), 2*(radius+offset), angle=0.0, theta1=start, theta2=end, color=color, linewidth=self.line_width)
            return arc, arc_out, start

        def get_circle(self, o, r, color='black'):
            circle = Arc([o.real, o.imag], 2*r, 2*r, angle=0.0, theta1=0, theta2=360, color=color, linewidth=self.line_width)
            return circle

        def get_line(self, p1, p2, color='black'):
            line = Line2D([p1.real, p2.real], [p1.imag, p2.imag], marker='o', markersize=self.mark_size, linestyle='-', color=color, linewidth=self.line_width)
            return line
        
        ## Set Postion Of Leg ##  
        def update(self, theta=np.deg2rad(17.0), beta=0, O=np.array([0, 0])):
            self.O = np.array(O) # origin of leg in world coordinate
            self.calculate(theta, 0, 0, beta, 0, 0)
            # four rims (rim, start point, center)
            self.set_rim(self.upper_rim_r, self.linkleg.F_c, self.linkleg.O1_c)
            self.set_rim(self.upper_rim_l, self.linkleg.H, self.linkleg.O1)
            self.set_rim(self.lower_rim_r, self.linkleg.G, self.linkleg.O2_c)
            self.set_rim(self.lower_rim_l, self.linkleg.F, self.linkleg.O2)
            # five joints on the rims   (joint, center)
            self.set_joint(self.upper_joint_r, self.linkleg.H_c)
            self.set_joint(self.upper_joint_l, self.linkleg.H)
            self.set_joint(self.lower_joint_r, self.linkleg.F_c)
            self.set_joint(self.lower_joint_l, self.linkleg.F)
            self.set_joint(self.G_joint, self.linkleg.G)
            # six bars  (bar, point1, point2)
            self.set_bar(self.OB_bar_r, 0, self.linkleg.B_c)
            self.set_bar(self.OB_bar_l, 0, self.linkleg.B)
            self.set_bar(self.AE_bar_r, self.linkleg.A_c, self.linkleg.E)
            self.set_bar(self.AE_bar_l, self.linkleg.A, self.linkleg.E)
            self.set_bar(self.CD_bar_r, self.linkleg.C_c, self.linkleg.D_c)
            self.set_bar(self.CD_bar_l, self.linkleg.C, self.linkleg.D)
            
        def set_rim(self, rim, p1, o):  # rim, start point, center
            start = np.angle(p1-o, deg=True) 
            for arc in rim.arc: # inner & outer arcs
                arc.set_center([o.real, o.imag] + self.O)    # center(x, y)
                arc.set_angle( start - rim.start )    # rotate angle (degree)
                
        def set_joint(self, joint, center): # joint, center
            joint.set_center([center.real, center.imag] + self.O)
            
        def set_bar(self, bar, p1, p2): # bar, point1, point2
            bar.set_data([p1.real, p2.real] + self.O[0], [p1.imag, p2.imag] + self.O[1])
            

    class GroundLine:
        def __init__(self, ax, length): # length: m
            self.unit = 0.01    # 0.01 m
            self.length = length
            self.segments = int(length/self.unit)   # how many s_lines
            self.l_line, = ax.plot([], [], linestyle='-.', color='darkgreen', linewidth=1)  # long line (ground)
            self.s_lines = []
            for _ in range(self.segments):  # n short lines (underground) distrbuted in ground
                line, = ax.plot([], [], '-.', color='darkgreen', linewidth=0.3)
                self.s_lines.append(line)
            
        def plot_ground(self, height):  # height: m
            self.l_line.set_data([-self.length/2, self.length/2], [height, height])
            # underground line: slight tilt ( (0, 0) to (unit, -0.5*unit) )
            x1 = np.linspace(-self.length/2, self.length/2, self.segments)
            y1 = np.linspace(height, height, self.segments)
            x2 = x1 + self.unit
            y2 = y1 - self.unit*0.5
            for i, line in enumerate(self.s_lines):
                line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            return [self.l_line] + self.s_lines

    # initialization of plot 
    def plot_init(self): 
        self.ax.clear()  # clear plot
        # plot setting
        self.ax.set_aspect('equal')  # 座標比例相同
        self.ax.set_xlim(-self.plot_width, self.plot_width)
        self.ax.set_ylim(-self.plot_width, self.plot_width)
        # initialize all graphics 
        self.ani_leg.get_shape(self.theta_list[0], self.beta_list[0])   # initial pose of leg
        self.center_line, = self.ax.plot([], [], linestyle='--', color='blue', linewidth=1)   # center line
        self.joint_points = [ self.ax.plot([], [], 'ko', markersize=self.mark_size)[0] for _ in range(5) ]   # five dots at the center of joints
        self.ground_line = self.GroundLine(self.ax, self.ground_length)   # ground line
        # add leg part to the plot
        self.leg_list = []   # store each part of leg
        for key, value in self.ani_leg.__dict__.items():
            if "rim" in key:
                self.ax.add_patch(value.arc[0])
                self.ax.add_patch(value.arc[1])
                self.leg_list.append(value.arc[0])
                self.leg_list.append(value.arc[1])
            elif "joint" in key:
                self.ax.add_patch(value)
                self.leg_list.append(value)
            elif "bar" in key:
                self.ax.add_line(value)
                self.leg_list.append(value)
        return self.leg_list
           
    # plot each frame
    def plot_update(self, frame):
        theta = self.theta_list[frame]
        beta = self.beta_list[frame]

        ## Update New Position ##
        ani_leg = self.ani_leg
        ani_leg.update(theta, beta) # update new position of leg 

        # plot center line
        center_G = ani_leg.G_joint.get_center()
        self.center_line.set_data([center_G[0], -center_G[0]], [center_G[1], -center_G[1]])
        
        # plot five dots at the center of joints
        for i, circle in enumerate([ani_leg.upper_joint_r, ani_leg.upper_joint_l, ani_leg.lower_joint_r, ani_leg.lower_joint_l, ani_leg.G_joint]):
            center = circle.get_center()
            self.joint_points[i].set_data([center[0]], [center[1]])
        
        # plot ground line  
        ground_height = self.contact_height(theta, beta)
        ground_list = self.ground_line.plot_ground(ground_height)
        
        # set window range to make ground don't move.
        if ground_height != -100:
            # self.ax.set_xlim(-self.plot_width, self.plot_width)
            self.ax.set_ylim(ground_height - 0.5*self.plot_width, ground_height + 1.5*self.plot_width)
            self.ax.figure.canvas.draw()
        
        return self.leg_list + [self.center_line] + self.joint_points + ground_list
    
    # find lowest point of leg
    def contact_height(self, theta, beta):
        ani_leg = self.ani_leg
        self.contact_map.mapping(theta, beta)
        contact_type = self.contact_map.rim
        in_the_air = False
        if contact_type == 1:
            lowest_part = ani_leg.upper_rim_l.arc[1]
        elif contact_type == 2:
            lowest_part = ani_leg.lower_rim_l.arc[1]       
        elif contact_type == 3:
            lowest_part = ani_leg.G_joint       
        elif contact_type == 4:
            lowest_part = ani_leg.lower_rim_r.arc[1]       
        elif contact_type == 5:
            lowest_part = ani_leg.upper_rim_r.arc[1] 
        else:   # none
            in_the_air = True
        # return ground_height  
        if in_the_air:
            return -100    
        else:
            return lowest_part.get_center()[1] - lowest_part.get_height()/2  
        
    ## Parameters Setting ##
    def setting(self, fig_size=10, plot_width=0.4, ground_length=0.6, interval=20):
        self.fig_size = fig_size
        self.plot_width = plot_width
        self.ground_length = ground_length
        self.interval = interval
        
        # self.mark_size = int(fig_size / 3)
        # self.line_width = int(fig_size / 10) + 1
        self.mark_size = 2.0
        self.line_width = 1.0
        
    # create animation file
    def create_ani(self, theta_list, beta_list, file_name, type='.gif', show=True):
        self.theta_list = theta_list
        self.beta_list = beta_list
        frames = len(self.theta_list)
        fig, self.ax = plt.subplots( figsize=(self.fig_size, self.fig_size) )
        print("Start")               
        if frames > 1:
            ani = FuncAnimation(fig, self.plot_update, frames=frames, interval=self.interval, init_func=self.plot_init, blit=True)
            if show:
                plt.show()
            ani.save(file_name + type, fps=1000/self.interval)
        else:
            self.plot_init()
            self.plot_update(0)
            plt.savefig(file_name + '.jpg')
            plt.close()
        
    # plot leg on one fig given from user
    def plot_one(self, theta=np.deg2rad(17.0), beta=0, O=np.array([0, 0]), ax=None): 
        if ax is None:
            fig, ax = plt.subplots()
        # plot setting
        ax.set_aspect('equal')  # 座標比例相同
        # initialize all graphics 
        self.ani_leg.get_shape(theta, beta)   # initial pose of leg
        self.ani_leg.update(theta, beta, O)  # update to apply displacement of origin of leg.
        self.center_line, = ax.plot([], [], linestyle='--', color='blue', linewidth=1)   # center line
        self.joint_points = [ ax.plot([], [], 'ko', markersize=self.mark_size)[0] for _ in range(5) ]   # five dots at the center of joints
        # add leg part to the plot
        for key, value in self.ani_leg.__dict__.items():
            if "rim" in key:
                ax.add_patch(value.arc[0])
                ax.add_patch(value.arc[1])
            elif "joint" in key:
                ax.add_patch(value)
            elif "bar" in key:
                ax.add_line(value)
        # joint points
        ani_leg = self.ani_leg
        for i, circle in enumerate([ani_leg.upper_joint_r, ani_leg.upper_joint_l, ani_leg.lower_joint_r, ani_leg.lower_joint_l, ani_leg.G_joint]):
            center = circle.get_center()
            self.joint_points[i].set_data([center[0]], [center[1]])
            
        return ax
        
if __name__ == '__main__':
    file_name = 'LegAnimation_demo'
    frames = 100
    theta_list = [ ( 17 + ( np.sin(0.2*theta) + 1 ) * (160-17)/2 ) *np.pi/180 for theta in range(frames) ] # 17~160 degree in sine wave
    beta_list = [ beta *np.pi/180 for beta in range(frames) ] 
    
    Animation = SingleLeg_Animation()  # rad
    Animation.setting(fig_size=10, plot_width=0.4, ground_length=0.6, interval=20)
    Animation.create_ani(theta_list, beta_list, file_name, type='.mp4', show=True)
    ax = Animation.plot_one()
    plt.show()

    