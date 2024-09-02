#!/usr/bin/env python3
import rospy
from corgi_ros_bridge.msg import RobotStamped
import numpy as np

from quad_mpc_3df import *
from cpg import *
from swing import *
from filter import *
import matplotlib.pyplot as plt
import yaml
import threading

# default param.
closed_loop = 1
standup_duration = 3
standup_idle_duration = 10
step_length = 0.2
step_height = 0.03
stance_height = 0.2
T_sw = 1
cycle = 10
gait_tolx_1 = 0.024
gait_tolx_2 = 0.024
pose_x_pid = np.array([[0.3], [0.2], [0]])
pose_z_pid = np.array([[0.3], [0.2], [0]])
orient_p_pid = np.array([[0.3], [0.1], [0]])
filter_x_freq = 5
filter_z_freq = 10
filter_pitch_freq = 10
noise_mode = 0

class ImpedanceParam:
    def __init__(self):
        self.M = np.array([0.652, 0.652])  # Mx, My
        self.K0 = np.array([30000, 30000])  # Kx, Ky
        self.D = np.array([400, 400])  # Dx, Dy
        self.K_pid_x = np.array([2000, 1800, 50])
        self.K_pid_y = np.array([2000, 1800, 50])
        
        self.K0 = np.array([1e8, 1e8])  # Kx, Ky
        self.K_pid_x = np.array([0, 0, 0])
        self.K_pid_y = np.array([0, 0, 0])


class LegState:
    def __init__(self):
        self.phi = np.array([[0], [0]])
        self.tb = np.array([[np.deg2rad(17)], [0]])
        self.pose = np.array([[0], [-0.1]])
        self.force = np.array([[0], [0]])
        self.imp_param = ImpedanceParam()


class RobotState:
    def __init__(self):
        self.pose = np.zeros([3, 1])
        self.raw_pose = np.zeros([3, 1])
        self.filtered_pose = np.zeros([3,1])
        self.qwxyz = np.zeros([4, 1])
        self.twist = np.zeros([6, 1])  # [w;v]
        self.legs = [LegState() for _ in range(4)]


class logger:
    def __init__(self, dt=0.01):
        self.xs = []
        self.zs = []
        self.ps = []  # pitch
        self.xds = []
        self.zds = []
        self.pds = []
        self.t = 0
        self.dt = dt

    def update_data(self, x, z, p, xd, zd, pd):
        self.xs.append(x)
        self.zs.append(z)
        self.ps.append(p)
        self.xds.append(xd)
        self.zds.append(zd)
        self.pds.append(pd)
        self.t += self.dt

    def savefig(self):
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(4, 1, constrained_layout=True, figsize=(10, 15))
        # ts = np.arange(0, self.t - self.dt, self.dt)
        ts = np.linspace(0, self.t, len(self.xs))

        ax[0].set_title("Body X-dir tracking")
        ax[0].plot(ts, self.xs, label="robot_state")
        ax[0].plot(ts, self.xds, label="reference")
        ax[0].legend()
        ax[0].grid(True, linewidth=0.5, color="gray")

        ax[1].set_title("Body Z-dir tracking")
        ax[1].plot(ts, self.zs, label="robot_state")
        ax[1].plot(ts, self.zds, label="reference")
        ax[1].legend()
        ax[1].grid(True, linewidth=0.5, color="gray")

        ax[2].set_title("Body pitch tracking")
        ax[2].plot(ts, self.ps, label="robot_state")
        ax[2].plot(ts, self.pds, label="reference")
        ax[2].legend()
        ax[2].grid(True, linewidth=0.5, color="gray")

        ax[3].set_title("Tracking Error")
        ax[3].plot(np.array(self.xds) - np.array(self.xs), label="error_x")
        ax[3].plot(np.array(self.zds) - np.array(self.zs), label="error_z")
        ax[3].plot(np.array(self.pds) - np.array(self.ps), label="error_pitch")
        ax[3].legend()
        ax[3].grid(True, linewidth=0.5, color="gray")

        # Adjust the spacing between subplots
        # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2)

        fig.savefig("/root/corgi_rosws/test.png", dpi=200, format="png")


class VectorPIDController:
    def __init__(self, Kp, Ki, Kd, T, setpoint):
        """
        Initialize the PID controller with coefficients and sampling time.
        :param Kp: Proportional gain (array)
        :param Ki: Integral gain (array)
        :param Kd: Derivative gain (array)
        :param T: Sampling time
        :param setpoint: Desired value of the process variable (array)
        """
        self.Kp = np.array(Kp, dtype=np.float64)
        self.Ki = np.array(Ki, dtype=np.float64)
        self.Kd = np.array(Kd, dtype=np.float64)
        self.T = float(T)  # Ensure T is a float
        self.setpoint = np.array(setpoint, dtype=np.float64)

        # Initialize delta terms for bilinear transformation with the correct dtype
        self.last_error = np.zeros_like(self.setpoint, dtype=np.float64)
        self.integral = np.zeros_like(self.setpoint, dtype=np.float64)
        self.first_elapsed = False

    def update(self, measurement):
        """
        Update the PID controller.
        :param measurement: Current measurement of the process variable (array)
        :return: Control variable (array)
        """
        error = self.setpoint - np.array(measurement, dtype=np.float64)

        # Integral with bilinear (trapezoidal) approximation
        # self.integral += 0.5 * self.T * (error + self.last_error)

        self.integral += self.T * error

        # Derivative with bilinear approximation
        if self.first_elapsed:
            derivative = (error - self.last_error) / self.T
        else:
            derivative = 0

        # Update last_error for next derivative calculation
        self.last_error = error
        self.first_elapsed = True

        # PID output
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class WalkGait:
    def __init__(self, dt):
        self.step_length = step_length
        self.step_height = step_height
        self.stance_height = stance_height
        self.T_sw = T_sw
        self.T_st = 3 * self.T_sw

        self.dt = dt
        self.cycle = cycle
        self.cycle_cnt = 0
        self.duration = self.cycle * (4 * self.T_sw)
        self.N_iter = math.ceil(self.duration / self.dt)
        self.standup_duration = standup_duration
        self.standup_iter = math.ceil(self.standup_duration / self.dt)

        self.vx = 1 * self.step_length / (4 * self.T_sw)
        self.wy = 0
        self.v1_x = 0.0
        self.v2_x = 0.0
        self.v3_x = 0.0
        self.gait_tolx_1 = gait_tolx_1
        self.gait_tolx_2 = gait_tolx_2

        # initial condition
        self.p0 = np.array([[-0.02], [0], [self.stance_height]])
        self.q0_wxyz = np.array([[1], [0], [0], [0]])
        
        self.init_dx = 1.0/4 * self.step_length
        com_shift = 0.01
        # self.init_dx = 0.05
        self.init_leg_pose = [
            np.array([-self.init_dx+com_shift, -self.stance_height]),
            np.array([self.init_dx+com_shift, -self.stance_height]),
            np.array([-self.init_dx+com_shift, -self.stance_height]),
            np.array([self.init_dx+com_shift, -self.stance_height]),
        ]
        
        # For openloop mode
        self.X_ref_rt_prev = None
        self.st_rs_prev = []
        
        self.stand_xfttraj = []
        self.stand_yfttraj = []

        self.X_ref = []
        self.xi_ref = []

    def setup_reference_traj(self):
        # Generate Reference trajectory
        p0_ref = self.p0
        R0_ref = quat2rotm(self.q0_wxyz)

        w0_ref = np.array([[0], [self.wy], [0]])
        v0_ref = np.array([[self.vx], [0], [0]])

        X0_ref = np.block([[R0_ref, p0_ref], [np.zeros([1, 3]), 1]])  # in SE3 form
        xid_ref = np.block([[w0_ref], [v0_ref]])  # reference body twist

        X_refs = [X0_ref.copy() for _ in range(self.N_iter)]
        xi_refs = [xid_ref.copy() for _ in range(self.N_iter)]
        X = X0_ref

        for i in range(1, self.N_iter):
            xid_ref_rt = xid_ref
            Xi_ = np.block([[skew(xid_ref_rt[:3]), xid_ref_rt[3:]], [0, 0, 0, 0]])
            X = X @ sp.linalg.expm(Xi_ * self.dt)
            X_refs[i] = X
            xi_refs[i] = xid_ref_rt.copy()

        self.X_ref = X_refs
        self.xi_ref = xi_refs
        
        print("Reference trajectory iteration: ", self.N_iter)
        print("X_refs: ", len(X_refs))

        self.stand_yfttraj = np.linspace(-0.1, -self.stance_height, self.standup_iter)
        self.stand_xfttraj = [np.linspace(0, self.init_leg_pose[i][0], self.standup_iter) for i in range(4)]

    def setup_reference_traj_tol(self):
        tolx_1 = self.gait_tolx_1
        tolx_2 = self.gait_tolx_2
        # Generate Reference trajectory
        p0_ref = self.p0
        R0_ref = quat2rotm(self.q0_wxyz)

        w0_ref = np.array([[0], [self.wy], [0]])
        v0_ref = np.array([[self.vx], [0], [0]])

        X0_ref = np.block([[R0_ref, p0_ref], [np.zeros([1, 3]), 1]])  # in SE3 form
        xid_ref = np.block([[w0_ref], [v0_ref]])  # reference body twist

        X_refs = [X0_ref.copy() for _ in range(self.N_iter)]
        xi_refs = [xid_ref.copy() for _ in range(self.N_iter)]
        X = X0_ref
        
        self.v1_x = (self.init_dx + tolx_1) / self.T_sw
        self.v2_x = (self.step_length - 2*self.init_dx + tolx_2 - tolx_1)/(2*self.T_sw)
        self.v3_x = (2*self.init_dx + tolx_1 - tolx_2) / (2*self.T_sw)
        xid1_ref = np.block([[w0_ref], [np.array([[self.v1_x], [0], [0]]) ]])
        xid2_ref = np.block([[w0_ref], [np.array([[self.v2_x], [0], [0]]) ]])
        xid3_ref = np.block([[w0_ref], [np.array([[self.v3_x], [0], [0]]) ]])
        
        # t_ = self.dt
        t_ = 0
        for i in range(1, self.N_iter):
            cpg_cnt = math.floor(t_/self.T_sw)
            if cpg_cnt == 0:
                xid_ref_rt = xid1_ref
            else:
                """ if cpg_cnt %4 == 0 or cpg_cnt %4 == 3:
                    xid_ref_rt = xid1_ref
                else:
                    xid_ref_rt = xid2_ref """
                    
                if cpg_cnt %4 == 0 or cpg_cnt %4 == 3:
                    xid_ref_rt = xid3_ref
                else:
                    xid_ref_rt = xid2_ref
            
            Xi_ = np.block([[skew(xid_ref_rt[:3]), xid_ref_rt[3:]], [0, 0, 0, 0]])
            X = X @ sp.linalg.expm(Xi_ * self.dt)
            X_refs[i] = X
            xi_refs[i] = xid_ref_rt.copy()
            
            t_ += self.dt
        

        self.X_ref = X_refs
        self.xi_ref = xi_refs
        
        print("Reference trajectory iteration: ", self.N_iter)
        print("X_refs: ", len(X_refs))

        self.stand_yfttraj = np.linspace(-0.1, -self.stance_height, self.standup_iter)
        self.stand_xfttraj = [np.linspace(0, self.init_leg_pose[i][0], self.standup_iter) for i in range(4)]


class Corgi_mpc:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.mpc_param_ = None
        self.mpc_controller = None
        self.mpc_setup()

        self.gait = WalkGait(dt)
        self.gait.setup_reference_traj()

        # ROS setup
        self.timer = rospy.Timer(rospy.Duration(dt), self.walk_loop)
        self.state = RobotState()
        self.state_sub = rospy.Subscriber("/robot/state", RobotStamped, self.stateCallback)
        self.force_pub = rospy.Publisher("/robot/command", RobotStamped, queue_size=1024)

        # impedance param for pure position control
        self.pos_param = ImpedanceParam()
        self.pos_param.K0 = np.array([1e6, 1e6])
        self.pos_param.K_pid_x = np.zeros(3)
        self.pos_param.K_pid_y = np.zeros(3)
        # normal impedance control param
        self.imp_param = ImpedanceParam()

        self.fsm = "idle"
        self.standup_time = 1
        self.stand_fttraj = []
        self.stand_cnt = 0

        # CPG
        self.cpg = CPG(dt)
        # swing planner
        self.swp = SwingLegPlanner(0.01, self.gait.T_sw, self.gait.T_st)
        self.walk_swing_profiles = None
        self.swing_comp = np.array([0,0,0,0], dtype=np.float64)
        self.swing_landpt = []

        # static transform (hip2body)
        w = 400 * 0.001
        wb = 440 * 0.001
        self.bhip = [
            np.array([wb / 2, w / 2, 0]),
            np.array([wb / 2, -w / 2, 0]),
            np.array([-wb / 2, -w / 2, 0]),
            np.array([-wb / 2, w / 2, 0]),
        ]

        self.ref_cnt = 0
        self.if_init = False
        self.if_track_finished = False

        """ # noise version
        # self.body_pos_pid = VectorPIDController([0.2, 0, 0.2], [0.05, 0, 0.05], [0.0, 0, 0.0], dt, setpoint=[0, 0, 0])
        # self.body_pos_pid = VectorPIDController([0.3, 0, 0.3], [0.2, 0, 0.2], [0.0, 0, 0.0], dt, setpoint=[0, 0, 0])
        # self.body_pitch_pid = VectorPIDController(
        #     np.array([0.3]),
        #     np.array([0.1]),
        #     np.array([0]),
        #     dt,
        #     setpoint=np.array([0]),
        # ) """
        self.close_loop = closed_loop
        self.body_pos_pid = VectorPIDController([pose_x_pid[0], 0, pose_z_pid[0]], \
                                                [pose_x_pid[1], 0, pose_z_pid[1]], \
                                                [pose_x_pid[2], 0, pose_z_pid[2]], \
                                                dt, setpoint=[0, 0, 0])
        
        self.body_pitch_pid = VectorPIDController([orient_p_pid[0]], [orient_p_pid[1]], [orient_p_pid[2]], \
                                                  dt, setpoint=[0])
        self.feedback_filter_x = LowPassFilter(filter_x_freq, 1/self.dt, order=6)
        self.feedback_filter_z = LowPassFilter(filter_z_freq, 1/self.dt, order=6)
        self.feedback_filter_p = LowPassFilter(filter_pitch_freq, 1/self.dt, order=6)
        self.logger = logger()

    def initialize(self):
        self.mpc_setup()
        self.mpc_controller = MPC(self.mpc_param_)
        self.if_init = True

    def mpc_setup(self):
        Ns = 12  # state dim.
        Nu = 12  # input dim.
        Nt = 12  # horizon
        dt = self.dt  # control frequency
        Tsim = 4
        Nsim = math.ceil(Tsim / dt)
        xmax = np.full((Ns, 1), np.inf)
        xmin = np.full((Ns, 1), -np.inf)
        umax = np.full((Nu, 1), 4000)
        umin = np.full((Nu, 1), -4000)
        self.mpc_param_ = param(Ns, Nu, Nt, dt, xmax, xmin, umax, umin)

    def stateCallback(self, data):
        if noise_mode == 1:
            noise_x = np.random.normal(data.pose.position.x, 0.0001, 1)[0]
            noise_y = np.random.normal(data.pose.position.y, 0.00, 1)[0]
            noise_z = np.random.normal(data.pose.position.z, 0.002, 1)[0]
            filtered_x = self.feedback_filter_x.apply_filter(noise_x)
            filtered_z = self.feedback_filter_z.apply_filter(noise_z)
            self.state.raw_pose = np.array([[data.pose.position.x], [noise_y], [noise_z]])
            self.state.filtered_pose = np.array([[filtered_x], [noise_y], [filtered_z]])
            self.state.pose = np.array([[filtered_x], [noise_y], [filtered_z]])
        else:
            self.state.pose = np.array([[data.pose.position.x], [data.pose.position.y], [data.pose.position.z]])
        
        self.state.qwxyz = np.array(
            [
                [data.pose.orientation.w],
                [data.pose.orientation.x],
                [data.pose.orientation.y],
                [data.pose.orientation.z],
            ]
        )
        self.state.twist = np.array(
            [
                [data.twist.angular.x],
                [data.twist.angular.y],
                [data.twist.angular.z],
                [data.twist.linear.x],
                [data.twist.linear.y],
                [data.twist.linear.z],
            ]
        )

        self.state.legs[0].tb = np.array([[data.A_LF.theta], [data.A_LF.beta]])
        self.state.legs[0].phi = np.array([[data.A_LF.motor_r.angle], [data.A_LF.motor_l.angle]])
        self.state.legs[0].pose = np.array([[data.A_LF.force.pose_x], [data.A_LF.force.pose_y]])
        self.state.legs[0].force = np.array([[data.A_LF.force.force_x], [data.A_LF.force.force_y]])
        self.state.legs[0].imp_param.M = np.array([data.A_LF.impedance.M_x, data.A_LF.impedance.M_y])
        self.state.legs[0].imp_param.K0 = np.array([data.A_LF.impedance.K0_x, data.A_LF.impedance.K0_y])
        self.state.legs[0].imp_param.D = np.array([data.A_LF.impedance.D_x, data.A_LF.impedance.D_y])
        self.state.legs[0].imp_param.K_pid_x = np.array(
            [
                data.A_LF.impedance.adaptive_kp_x,
                data.A_LF.impedance.adaptive_ki_x,
                data.A_LF.impedance.adaptive_kd_x,
            ]
        )
        self.state.legs[0].imp_param.K_pid_y = np.array(
            [
                data.A_LF.impedance.adaptive_kp_y,
                data.A_LF.impedance.adaptive_ki_y,
                data.A_LF.impedance.adaptive_kd_y,
            ]
        )

        self.state.legs[1].tb = np.array([[data.B_RF.theta], [data.B_RF.beta]])
        self.state.legs[1].phi = np.array([[data.B_RF.motor_r.angle], [data.B_RF.motor_l.angle]])
        self.state.legs[1].pose = np.array([[data.B_RF.force.pose_x], [data.B_RF.force.pose_y]])
        self.state.legs[1].force = np.array([[data.B_RF.force.force_x], [data.B_RF.force.force_y]])
        self.state.legs[1].imp_param.M = np.array([data.B_RF.impedance.M_x, data.B_RF.impedance.M_y])
        self.state.legs[1].imp_param.K0 = np.array([data.B_RF.impedance.K0_x, data.B_RF.impedance.K0_y])
        self.state.legs[1].imp_param.D = np.array([data.B_RF.impedance.D_x, data.B_RF.impedance.D_y])
        self.state.legs[1].imp_param.K_pid_x = np.array(
            [
                data.B_RF.impedance.adaptive_kp_x,
                data.B_RF.impedance.adaptive_ki_x,
                data.B_RF.impedance.adaptive_kd_x,
            ]
        )
        self.state.legs[1].imp_param.K_pid_y = np.array(
            [
                data.B_RF.impedance.adaptive_kp_y,
                data.B_RF.impedance.adaptive_ki_y,
                data.B_RF.impedance.adaptive_kd_y,
            ]
        )

        self.state.legs[2].tb = np.array([[data.C_RH.theta], [data.C_RH.beta]])
        self.state.legs[2].phi = np.array([[data.C_RH.motor_r.angle], [data.C_RH.motor_l.angle]])
        self.state.legs[2].pose = np.array([[data.C_RH.force.pose_x], [data.C_RH.force.pose_y]])
        self.state.legs[2].force = np.array([[data.C_RH.force.force_x], [data.C_RH.force.force_y]])
        self.state.legs[2].imp_param.M = np.array([data.C_RH.impedance.M_x, data.C_RH.impedance.M_y])
        self.state.legs[2].imp_param.K0 = np.array([data.C_RH.impedance.K0_x, data.C_RH.impedance.K0_y])
        self.state.legs[2].imp_param.D = np.array([data.C_RH.impedance.D_x, data.C_RH.impedance.D_y])
        self.state.legs[2].imp_param.K_pid_x = np.array(
            [
                data.C_RH.impedance.adaptive_kp_x,
                data.C_RH.impedance.adaptive_ki_x,
                data.C_RH.impedance.adaptive_kd_x,
            ]
        )
        self.state.legs[2].imp_param.K_pid_y = np.array(
            [
                data.C_RH.impedance.adaptive_kp_y,
                data.C_RH.impedance.adaptive_ki_y,
                data.C_RH.impedance.adaptive_kd_y,
            ]
        )

        self.state.legs[3].tb = np.array([[data.D_LH.theta], [data.D_LH.beta]])
        self.state.legs[3].phi = np.array([[data.D_LH.motor_r.angle], [data.D_LH.motor_l.angle]])
        self.state.legs[3].pose = np.array([[data.D_LH.force.pose_x], [data.D_LH.force.pose_y]])
        self.state.legs[3].force = np.array([[data.D_LH.force.force_x], [data.D_LH.force.force_y]])
        self.state.legs[3].imp_param.M = np.array([data.D_LH.impedance.M_x, data.D_LH.impedance.M_y])
        self.state.legs[3].imp_param.K0 = np.array([data.D_LH.impedance.K0_x, data.D_LH.impedance.K0_y])
        self.state.legs[3].imp_param.D = np.array([data.D_LH.impedance.D_x, data.D_LH.impedance.D_y])
        self.state.legs[3].imp_param.K_pid_x = np.array(
            [
                data.D_LH.impedance.adaptive_kp_x,
                data.D_LH.impedance.adaptive_ki_x,
                data.D_LH.impedance.adaptive_kd_x,
            ]
        )
        self.state.legs[3].imp_param.K_pid_y = np.array(
            [
                data.D_LH.impedance.adaptive_kp_y,
                data.D_LH.impedance.adaptive_ki_y,
                data.D_LH.impedance.adaptive_kd_y,
            ]
        )

    def walk_loop(self, event):
        if self.fsm == "idle":
            pass

        elif self.fsm == "standup":
            """ if np.abs(self.state.pose[2] - self.gait.X_ref[0][2, 3]) < 0.005:
                print("switch to walk mode")
                self.fsm = "walk"
                self.cpg.switchGait("walk", self.gait.T_sw)

            else:
                st_rs_ = [
                    np.array([self.gait.stand_xfttraj[i][self.stand_cnt], 0, self.gait.stand_yfttraj[self.stand_cnt]])
                    # np.array([0, 0, self.gait.stand_yfttraj[self.stand_cnt]])
                    for i in range(4)
                ]
                if self.stand_cnt < len(self.gait.stand_yfttraj) - 1:
                    self.stand_cnt += 1

                print("[stand] ", st_rs_[0])
                u_ = np.ones([12, 1]) * 0
                self.commandPublish(u_, st_rs_, self.pos_param) """
            st_rs_ = [
                np.array([self.gait.stand_xfttraj[i][self.stand_cnt], 0, self.gait.stand_yfttraj[self.stand_cnt]])
                for i in range(4)
            ]
            if self.stand_cnt < len(self.gait.stand_yfttraj) - 1:
                self.stand_cnt += 1
            else:
                print("switch to idle mode")
                self.fsm = "idle"
                self.gait.p0 = self.state.pose
                self.gait.setup_reference_traj_tol()
                self.cpg.switchGait("walk", self.gait.T_sw)
                self.swing_landpt = [self.state.pose.reshape(3) + self.bhip[i] + \
                                   np.array([self.state.legs[i].pose[0,0], 0, self.state.legs[i].pose[1,0]]) \
                                    for i in range(4)]
                # for open loop mode
                self.X_ref_rt_prev = self.gait.X_ref[0]
                self.st_rs_prev = [np.array([self.gait.init_leg_pose[i][0], 0, self.gait.init_leg_pose[i][1]]) 
                                   for i in range(4)]
                

            print("[stand] ", st_rs_[0])
            u_ = np.ones([12, 1]) * 0
            self.commandPublish(u_, st_rs_, self.pos_param)

        elif self.fsm == "walk":
            legs_contact, legs_duty, if_switch_leg = self.cpg.update()

            X_ref_rt = self.gait.X_ref[self.ref_cnt]
            xi_ref_rt = self.gait.xi_ref[self.ref_cnt : (self.ref_cnt + self.mpc_param_.Nt)]
            R_ = quat2rotm(self.state.qwxyz)
            X_se3_ = np.block([[R_, self.state.pose], [0, 0, 0, 1]])

            # Stance legs, compute optimal force for stance leg
            # Swing leg, compute continuous bezier trajectory
            st_rs_ = []  # footend pose w.r.t. body
            sw_rs_ = []
            sw_idx = -1
            sw_duty = 0
            for i in range(4):
                if legs_contact[i] == 1:
                    st_rs_.append(
                        self.bhip[i]
                        + np.array(
                            [
                                self.state.legs[i].pose[0, 0],
                                0,
                                self.state.legs[i].pose[1, 0],
                            ]
                        )
                    )
                else:
                    st_rs_.append(self.bhip[i] + np.array([0, 0, 0]))
                    sw_rs_.append(self.state.legs[i].pose.reshape(2))
                    sw_duty = legs_duty[i]
                    sw_idx = i

            st_u = self.mpc_controller.mpc(X_se3_, X_ref_rt, self.state.twist, xi_ref_rt, st_rs_)

            # Footend Position cmd generate by body pose, orientation tracking
            # translation
            self.body_pos_pid.setpoint = X_ref_rt[:3, 3].reshape(3)
            next_pose = self.state.pose + self.body_pos_pid.update(self.state.pose.reshape(3)).reshape(3, 1)
            # orientation
            desired_zyx = R.from_matrix(X_ref_rt[:3, :3]).as_euler("zyx")
            self.body_pitch_pid.setpoint = desired_zyx[1]
            current_zyx = R.from_matrix(R_).as_euler("zyx")
            current_pitch = current_zyx[1]
            d_pitch = self.body_pitch_pid.update(current_pitch)
            next_rmat = R.from_euler("zyx", [0, d_pitch[0], 0]).as_matrix() @ X_se3_[:3, :3]
            X_se3_next = np.block([[next_rmat, next_pose], [0, 0, 0, 1]])

            st_rs_next = []
            for i in range(4):
                if legs_contact[i] == 1:
                    if self.close_loop == 1:
                        st_rs_next.append(
                            (np.linalg.inv(X_se3_next) @ X_se3_ @ np.block([[st_rs_[i].reshape(3, 1)], [1]]))[:3, 0]
                            - self.bhip[i])
                    else:
                        # openloop mode
                        # print(i, self.st_rs_prev[i])
                        r_ = (np.linalg.inv(X_ref_rt)
                                @ self.X_ref_rt_prev
                                @ np.block([[self.st_rs_prev[i].reshape(3, 1)], [1]])
                            )[:3, 0]
                        st_rs_next.append(r_)
                        self.st_rs_prev[i] = r_

            # Swing Legs
            if if_switch_leg or self.walk_swing_profiles == None and sw_idx != -1:
                vxd_ = self.gait.step_length/(self.gait.T_sw*4)
                
                # print("cnt, ", self.ref_cnt, "sw_idx, ", sw_idx)
                if sw_idx == 2 or sw_idx == 1:
                    vx_ = self.gait.v2_x
                else:
                    vx_ = self.gait.v3_x
                # print(vx_)
                
                for i in range(4):
                    if i != sw_idx:
                        self.swing_comp[i] = self.swing_comp[i] + vx_ * self.gait.T_sw
                
                self.swing_comp[sw_idx] = 0.0
                
                w_liftoff_pt = self.state.pose.reshape(3) + self.bhip[sw_idx] + \
                                np.array([self.state.legs[sw_idx].pose[0,0], 0, self.state.legs[sw_idx].pose[1,0]])
                
                # w_touchdown_pt = w_liftoff_pt + np.array([self.gait.step_length, 0, 0])
                self.swing_landpt[sw_idx] = self.swing_landpt[sw_idx] + np.array([self.gait.step_length, 0, 0])
                w_touchdown_pt = self.swing_landpt[sw_idx]
                w_liftoff_pt = np.array([w_liftoff_pt[0], w_liftoff_pt[2]])
                w_touchdown_pt = np.array([w_touchdown_pt[0], w_touchdown_pt[2]])
                
                w_liftoff_pt[1] = 0
                w_touchdown_pt[1] = 0
                
                # print("w_liftoff_pt, ", w_liftoff_pt)
                # print("w_touchdown_pt, ", w_touchdown_pt)
                # print("--")
                
                sp_ = self.swp.solveSwingTrajectory(
                    w_liftoff_pt,
                    w_touchdown_pt,
                    self.gait.step_height,
                    np.array([-vx_, 0]),
                    np.array([-vx_, 0]),
                )
                
                self.walk_swing_profiles = sp_
                
            if sw_idx != -1:
                w_swing_pt = self.walk_swing_profiles.getFootendPoint(sw_duty)
                # b_swing_pt = w_swing_pt - (np.array(np.array([self.state.pose[0,0], self.state.pose[2,0]])) + np.array([self.bhip[sw_idx][0], self.bhip[sw_idx][2]]))
                b_swing_pt = w_swing_pt - (np.array(np.array([X_ref_rt[0,3], X_ref_rt[2,3]])) + np.array([self.bhip[sw_idx][0], self.bhip[sw_idx][2]]))
                sw_rs_next = np.array([[b_swing_pt[0]], [0], [b_swing_pt[1]]])
                # sw_rs_next = self.walk_swing_profiles.getFootendPoint(sw_duty)
                # sw_rs_next = np.array([[sw_rs_next[0]], [0], [sw_rs_next[1]]])
                if self.close_loop == 0:
                    self.st_rs_prev[sw_idx] = sw_rs_next.reshape(3)
                
            

            # Combine stance and swing leg command
            st_idx = 0
            rs_next = []
            for i in range(4):
                if legs_contact[i] == 1:
                    rs_next.append(st_rs_next[st_idx])
                    st_idx += 1
                else:
                    rs_next.append(sw_rs_next)
            
            self.commandPublish(st_u, rs_next, self.imp_param, X_ref_rt, xi_ref_rt[0])
            self.logger.update_data(
                X_se3_[0, 3],
                X_se3_[2, 3],
                current_pitch,
                X_ref_rt[0, 3],
                X_ref_rt[2, 3],
                desired_zyx[1],
            )

            if self.ref_cnt < len(self.gait.X_ref) - self.mpc_param_.Nt - 1:
                self.ref_cnt += 1
                self.X_ref_rt_prev = X_ref_rt
            elif not self.if_track_finished:
                print("tracking complete, ", self.ref_cnt)
                self.if_track_finished = True
            else:
                self.logger.savefig()
                self.fsm = "idle"

    def commandPublish(self, u, rs, imp_param, X_ref=None, xi_ref=None, seq=0):
        robot_msg = RobotStamped()
        # robot_msg.header.seq = seq
        # robot_msg.stamp = rospy.Time.now()
        robot_msg.msg_type = "force"

        robot_msg.A_LF.force.pose_x = rs[0][0]
        robot_msg.A_LF.force.pose_y = rs[0][2]
        robot_msg.A_LF.force.force_x = u[0]
        robot_msg.A_LF.force.force_y = u[2]

        robot_msg.A_LF.impedance.M_x = imp_param.M[0]
        robot_msg.A_LF.impedance.M_y = imp_param.M[1]
        robot_msg.A_LF.impedance.K0_x = imp_param.K0[0]
        robot_msg.A_LF.impedance.K0_y = imp_param.K0[1]
        robot_msg.A_LF.impedance.D_x = imp_param.D[0]
        robot_msg.A_LF.impedance.D_y = imp_param.D[1]
        robot_msg.A_LF.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        robot_msg.A_LF.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        robot_msg.A_LF.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        robot_msg.A_LF.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        robot_msg.A_LF.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        robot_msg.A_LF.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.B_RF.force.pose_x = rs[1][0]
        robot_msg.B_RF.force.pose_y = rs[1][2]
        robot_msg.B_RF.force.force_x = u[3]
        robot_msg.B_RF.force.force_y = u[5]
        robot_msg.B_RF.impedance.M_x = imp_param.M[0]
        robot_msg.B_RF.impedance.M_y = imp_param.M[1]
        robot_msg.B_RF.impedance.K0_x = imp_param.K0[0]
        robot_msg.B_RF.impedance.K0_y = imp_param.K0[1]
        robot_msg.B_RF.impedance.D_x = imp_param.D[0]
        robot_msg.B_RF.impedance.D_y = imp_param.D[1]
        robot_msg.B_RF.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        robot_msg.B_RF.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        robot_msg.B_RF.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        robot_msg.B_RF.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        robot_msg.B_RF.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        robot_msg.B_RF.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.C_RH.force.pose_x = rs[2][0]
        robot_msg.C_RH.force.pose_y = rs[2][2]
        robot_msg.C_RH.force.force_x = u[6]
        robot_msg.C_RH.force.force_y = u[8]
        robot_msg.C_RH.impedance.M_x = imp_param.M[0]
        robot_msg.C_RH.impedance.M_y = imp_param.M[1]
        robot_msg.C_RH.impedance.K0_x = imp_param.K0[0]
        robot_msg.C_RH.impedance.K0_y = imp_param.K0[1]
        robot_msg.C_RH.impedance.D_x = imp_param.D[0]
        robot_msg.C_RH.impedance.D_y = imp_param.D[1]
        robot_msg.C_RH.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        robot_msg.C_RH.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        robot_msg.C_RH.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        robot_msg.C_RH.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        robot_msg.C_RH.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        robot_msg.C_RH.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.D_LH.force.pose_x = rs[3][0]
        robot_msg.D_LH.force.pose_y = rs[3][2]
        robot_msg.D_LH.force.force_x = u[9]
        robot_msg.D_LH.force.force_y = u[11]
        robot_msg.D_LH.impedance.M_x = imp_param.M[0]
        robot_msg.D_LH.impedance.M_y = imp_param.M[1]
        robot_msg.D_LH.impedance.K0_x = imp_param.K0[0]
        robot_msg.D_LH.impedance.K0_y = imp_param.K0[1]
        robot_msg.D_LH.impedance.D_x = imp_param.D[0]
        robot_msg.D_LH.impedance.D_y = imp_param.D[1]
        robot_msg.D_LH.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        robot_msg.D_LH.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        robot_msg.D_LH.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        robot_msg.D_LH.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        robot_msg.D_LH.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        robot_msg.D_LH.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        if isinstance(X_ref, type(np.array([]))):
            qr = R.from_matrix(X_ref[:3, :3]).as_quat()
            robot_msg.pose.position.x = X_ref[0, 3]
            robot_msg.pose.position.y = X_ref[1, 3]
            robot_msg.pose.position.z = X_ref[2, 3]
            robot_msg.pose.orientation.x = qr[0]
            robot_msg.pose.orientation.y = qr[1]
            robot_msg.pose.orientation.z = qr[2]
            robot_msg.pose.orientation.w = qr[3]
            robot_msg.twist.angular.x = xi_ref[0]
            robot_msg.twist.angular.y = xi_ref[1]
            robot_msg.twist.angular.z = xi_ref[2]
            robot_msg.twist.linear.x = xi_ref[3]
            robot_msg.twist.linear.y = xi_ref[4]
            robot_msg.twist.linear.z = xi_ref[5]

        self.force_pub.publish(robot_msg)


def loadConfig(filepath="./config/yaml"):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    global closed_loop
    global standup_duration
    global standup_idle_duration
    global step_length
    global step_height
    global stance_height
    global T_sw
    global cycle
    global gait_tolx_1
    global gait_tolx_2
    global pose_x_pid
    global pose_z_pid
    global orient_p_pid
    global filter_x_freq
    global filter_z_freq
    global filter_pitch_freq
    global noise_mode
    closed_loop = data['closed_loop']
    standup_duration = data['standup_duration']
    standup_idle_duration = data['standup_idle_duration']
    step_length = data['step_length']
    step_height = data['step_height']
    stance_height = data['stance_height']
    T_sw = data['T_sw']
    cycle = data['cycle']
    gait_tolx_1 = data['gait_tolx_1']
    gait_tolx_2 = data['gait_tolx_2']
    pose_x_pid = data['pose_x_pid']
    pose_z_pid = data['pose_z_pid']
    orient_p_pid = data['orient_pitch_pid']
    filter_x_freq = data['filter_x_freq']
    filter_z_freq = data['filter_z_freq']
    filter_pitch_freq = data['filter_pitch_freq']
    noise_mode = data["noise_mode"]

def input_trigger(cobj):
    while True:
        mode = str(input("Mode Command [i]dle [w]alk [s]tand: "))
        if mode == "i":
            cobj.fsm = "idle"
        elif mode=="w":
            cobj.fsm = "walk"
        elif mode == "s":
            cobj.fsm = "standup"
        time.sleep(0.5)


if __name__ == "__main__":
    print("MPC started")
    rospy.init_node("corgi_mpc", anonymous=True)
    
    loadConfig(filepath="/root/corgi_rosws/src/corgi_mpc/config/config.yaml")
    

    cmpc = Corgi_mpc()
    cmpc.initialize()
    
    # terminal ui
    thread = threading.Thread(target=input_trigger, args=(cmpc,))
    thread.start()
    
    # cmpc.fsm = "standup"
    
    # a = int(input("aaa: "))
    # print(a)
    
    
    """ # plotter = RealTimePlot()
    # while True:
    #     plotter.update_plot(cmpc.state.raw_pose[0,0], cmpc.state.filtered_pose[0,0], cmpc.state.raw_pose[2,0], cmpc.state.filtered_pose[2,0])
    #     time.sleep(0.01) """
    

    rospy.spin()
