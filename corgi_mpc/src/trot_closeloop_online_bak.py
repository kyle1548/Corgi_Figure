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
import pandas as pd

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
filter_mode = 0
com_shift = 0
accel_L = 0


class ImpedanceParam:
    def __init__(self):
        self.M = np.array([0.652, 0.652])  # Mx, My
        self.K0 = np.array([5000, 5000])  # Kx, Ky
        self.D = np.array([400, 400])  # Dx, Dy
        self.K_pid_x = np.array([2000, 1800, 50])
        self.K_pid_y = np.array([2000, 1800, 50])

        # self.K0 = np.array([1e8, 1e8])  # Kx, Ky
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
        self.qwxyz = np.array([[1, 0, 0, 0]])
        self.raw_pose = np.zeros([3, 1])
        self.filtered_pose = np.zeros([3, 1])
        self.filtered_qwxyz = np.array([1, 0, 0, 0])
        self.twist = np.zeros([6, 1])  # [w;v]
        self.legs = [LegState() for _ in range(4)]
        self.x_raw = 0


class logger:
    def __init__(self, dt=0.01):
        self.seqs = []
        self.xs = []
        self.xs_raw = []
        self.zs = []
        self.ps = []  # pitch
        self.xds = []
        self.zds = []
        self.pds = []
        self.vxs = []
        self.vzs = []
        self.wps = []
        self.t = 0
        self.dt = dt

    def update_data(self, seq, x, z, p, xd, zd, pd, vx, vz, wp, x_raw=0):
        self.seqs.append(seq)
        self.xs.append(x)
        self.xs_raw.append(x_raw)
        self.zs.append(z)
        self.ps.append(p)
        self.xds.append(xd)
        self.zds.append(zd)
        self.pds.append(pd)
        self.vxs.append(vx)
        self.vzs.append(vz)
        self.wps.append(wp)
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

        fig.savefig("/home/biorola/test.png", dpi=200, format="png")

    def savedata(self, path="/home/biorola/walk_0620_1.csv"):
        data = {
            "seq": self.seqs,
            "x": self.xs,
            "z": self.zs,
            "p": self.ps,
            "xd": self.xds,
            "zd": self.zds,
            "pd": self.pds,
            "vx": self.vxs,
            "vz": self.vzs,
            "wp": self.wps,
            "x_raw": self.xs_raw,
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)


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


class TrotGait:
    def __init__(self, dt):
        self.step_length = step_length
        self.step_height = step_height
        self.stance_height = stance_height
        self.T_sw = T_sw
        self.T_st = T_sw

        self.dt = dt
        self.cycle = cycle
        self.cycle_cnt = 0
        self.duration = self.cycle * (2 * self.T_sw)
        self.N_iter = math.ceil(self.duration / self.dt)

        self.standup_duration = standup_duration
        self.standup_iter = math.ceil(self.standup_duration / self.dt)

        # initial condition
        self.p0 = np.array([[-0.02], [0], [self.stance_height]])
        self.q0_wxyz = np.array([[1], [0], [0], [0]])
        self.vx = self.step_length / (2 * self.T_sw)
        self.wy = 0

        self.init_dx = 1.0 / 4 * self.step_length
        self.init_leg_pose = [
            np.array([-self.init_dx + com_shift, -self.stance_height]),
            np.array([self.init_dx + com_shift, -self.stance_height]),
            np.array([-self.init_dx + com_shift, -self.stance_height]),
            np.array([self.init_dx + com_shift, -self.stance_height]),
        ]

        self.stand_xfttraj = []
        self.stand_yfttraj = []
        self.accel_xtraj = []

        self.X_ref = []
        self.xi_ref = []

    def setup_init_acceleration(self):
        v1 = self.vx
        T = 2 * accel_L / v1
        acc = v1 / T
        ts = np.arange(0, T + self.dt, self.dt)
        print("accel_L: ", accel_L)
        print("accel_T: ", T)
        print("accel: ", acc)

        self.accel_xtraj = [0.5 * acc * t_ for t_ in ts]

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
        print("step_L, ", self.step_length)
        print("T_sw, ", self.T_sw)
        print("vx, ", self.vx)

        self.stand_yfttraj = np.linspace(-0.1, -self.stance_height, self.standup_iter)
        self.stand_xfttraj = [
            np.linspace(0, self.init_leg_pose[i][0], self.standup_iter)
            for i in range(4)
        ]


class Corgi_mpc:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.mpc_param_ = None
        self.mpc_controller = None
        self.mpc_setup()

        self.gait = TrotGait(dt)
        self.gait.setup_reference_traj()

        # ROS setup
        self.timer = rospy.Timer(rospy.Duration(dt), self.trot_loop)
        # self.timer = rospy.Timer(rospy.Duration(dt), self.trot_loop)
        self.state = RobotState()
        self.state_sub = rospy.Subscriber(
            "/robot/state", RobotStamped, self.stateCallback
        )
        self.force_pub = rospy.Publisher(
            "/robot/command", RobotStamped, queue_size=1024
        )

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
        self.trot_swing_profiles = None
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
        self.seq = 0
        self.if_init = False
        self.if_track_finished = False

        self.motion_init = False

        self.close_loop = closed_loop
        self.X_ref_rt_prev = np.eye(4)
        self.st_rs_fb_prev = [
            np.array([self.gait.init_leg_pose[i][0], 0, self.gait.init_leg_pose[i][1]])
            for i in range(4)
        ]

        self.body_pos_pid = VectorPIDController(
            [pose_x_pid[0], 0, pose_z_pid[0]],
            [pose_x_pid[1], 0, pose_z_pid[1]],
            [pose_x_pid[2], 0, pose_z_pid[2]],
            dt,
            setpoint=[0, 0, 0],
        )

        self.body_pitch_pid = VectorPIDController(
            [orient_p_pid[0]], [orient_p_pid[1]], [orient_p_pid[2]], dt, setpoint=[0]
        )
        self.feedback_filter_x = LowPassFilter(filter_x_freq, 1 / self.dt, order=6)
        self.feedback_filter_z = LowPassFilter(filter_z_freq, 1 / self.dt, order=6)
        self.feedback_filter_p = LowPassFilter(filter_pitch_freq, 1 / self.dt, order=6)
        self.feedback_filter_vx = LowPassFilter(20, 1 / self.dt, order=6)
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
        self.seq = data.header.seq
        signal_x = 0
        signal_vx = 0
        signal_y = 0
        signal_z = 0
        if noise_mode == 1:
            # signal_x = np.random.normal(data.pose.position.x, 0.004, 1)[0]
            # signal_vx = np.random.normal(data.twist.linear.x, 0.05, 1)[0]
            vx_ = signal_vx
            # signal_vx = data.twist.linear.x
            vx_ = signal_vx + np.random.normal(0.05, 0.01, 1)[0]
            vx_ = self.feedback_filter_vx.apply_filter(vx_)
            signal_x = self.state.pose[0, 0] + vx_ * self.dt
            signal_y = np.random.normal(data.pose.position.y, 0.000, 1)[0]
            signal_z = np.random.normal(data.pose.position.z, 0.000, 1)[0]
        else:
            signal_x = data.pose.position.x
            signal_y = data.pose.position.y
            signal_z = data.pose.position.z

        if filter_mode == 1:
            signal_x = self.feedback_filter_x.apply_filter(signal_x)
            signal_z = self.feedback_filter_z.apply_filter(signal_z)

        vx_ = data.twist.linear.x
        vx_ = self.feedback_filter_vx.apply_filter(vx_)
        x_ = self.state.pose[0, 0] + self.dt * vx_

        self.state.pose = np.array([[signal_x], [signal_y], [signal_z]])
        # self.state.pose = np.array([[x_], [signal_y], [signal_z]])
        self.state.x_raw = data.pose.position.x

        if filter_mode == 0:
            self.state.qwxyz = np.array(
                [
                    [data.pose.orientation.w],
                    [data.pose.orientation.x],
                    [data.pose.orientation.y],
                    [data.pose.orientation.z],
                ]
            )
        else:
            if (
                np.linalg.norm(
                    [
                        data.pose.orientation.x,
                        data.pose.orientation.y,
                        data.pose.orientation.z,
                        data.pose.orientation.w,
                    ]
                )
                == 0
            ):
                ypr_ = np.array([0, 0, 0])
            else:
                ypr_ = R.from_quat(
                    [
                        data.pose.orientation.x,
                        data.pose.orientation.y,
                        data.pose.orientation.z,
                        data.pose.orientation.w,
                    ]
                ).as_euler("zyx")
            p_ = self.feedback_filter_p.apply_filter(ypr_[1])
            ypr_[1] = p_
            q_ = R.from_euler("zyx", ypr_).as_quat()
            self.state.qwxyz = np.array([q_[3], q_[0], q_[1], q_[2]]).reshape(4, 1)

        if closed_loop == 0:
            self.state.pose = self.X_ref_rt_prev[:3, 3].reshape(3, 1)
            self.state.qwxyz = np.array([1, 0, 0, 0]).reshape(4, 1)
            self.state.filtered_pose = np.array(
                [[data.pose.position.x], [data.pose.position.y], [data.pose.position.z]]
            )
            self.state.filtered_qwxyz = np.array(
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
        self.state.legs[0].phi = np.array(
            [[data.A_LF.motor_r.angle], [data.A_LF.motor_l.angle]]
        )
        self.state.legs[0].pose = np.array(
            [[data.A_LF.force.pose_x], [data.A_LF.force.pose_y]]
        )
        self.state.legs[0].force = np.array(
            [[data.A_LF.force.force_x], [data.A_LF.force.force_y]]
        )

        self.state.legs[1].tb = np.array([[data.B_RF.theta], [data.B_RF.beta]])
        self.state.legs[1].phi = np.array(
            [[data.B_RF.motor_r.angle], [data.B_RF.motor_l.angle]]
        )
        self.state.legs[1].pose = np.array(
            [[data.B_RF.force.pose_x], [data.B_RF.force.pose_y]]
        )
        self.state.legs[1].force = np.array(
            [[data.B_RF.force.force_x], [data.B_RF.force.force_y]]
        )

        self.state.legs[2].tb = np.array([[data.C_RH.theta], [data.C_RH.beta]])
        self.state.legs[2].phi = np.array(
            [[data.C_RH.motor_r.angle], [data.C_RH.motor_l.angle]]
        )
        self.state.legs[2].pose = np.array(
            [[data.C_RH.force.pose_x], [data.C_RH.force.pose_y]]
        )
        self.state.legs[2].force = np.array(
            [[data.C_RH.force.force_x], [data.C_RH.force.force_y]]
        )

        self.state.legs[3].tb = np.array([[data.D_LH.theta], [data.D_LH.beta]])
        self.state.legs[3].phi = np.array(
            [[data.D_LH.motor_r.angle], [data.D_LH.motor_l.angle]]
        )
        self.state.legs[3].pose = np.array(
            [[data.D_LH.force.pose_x], [data.D_LH.force.pose_y]]
        )
        self.state.legs[3].force = np.array(
            [[data.D_LH.force.force_x], [data.D_LH.force.force_y]]
        )

    def trot_loop(self, event):
        if self.fsm == "idle":
            pass

        elif self.fsm == "standup":
            st_rs_ = [
                np.array(
                    [
                        self.gait.stand_xfttraj[i][self.stand_cnt],
                        0,
                        self.gait.stand_yfttraj[self.stand_cnt],
                    ]
                )
                for i in range(4)
            ]
            if self.stand_cnt < len(self.gait.stand_yfttraj) - 1:
                self.X_ref_rt_prev = np.block(
                    [
                        [
                            np.eye(3),
                            np.array(
                                [0, 0, -self.gait.stand_yfttraj[self.stand_cnt]]
                            ).reshape(3, 1),
                        ],
                        [0, 0, 0, 1],
                    ]
                )
                self.stand_cnt += 1
            else:
                print("switch to idle mode")
                self.fsm = "idle"

                # for open loop mode
                self.X_ref_rt_prev = self.gait.X_ref[0]
                self.st_rs_prev = [
                    np.array(
                        [
                            self.gait.init_leg_pose[i][0],
                            0,
                            self.gait.init_leg_pose[i][1],
                        ]
                    )
                    for i in range(4)
                ]

            print("[stand] ", st_rs_[0])
            u_ = np.ones([12, 1]) * 0
            self.commandPublish(u_, st_rs_, self.pos_param)

        elif self.fsm == "trot":
            if self.motion_init == False:
                self.gait.p0 = self.state.pose
                self.gait.setup_reference_traj()
                self.swing_landpt = [
                    self.state.pose.reshape(3)
                    + self.bhip[i]
                    + np.array(
                        [
                            self.state.legs[i].pose[0, 0],
                            0,
                            self.state.legs[i].pose[1, 0],
                        ]
                    )
                    for i in range(4)
                ]
                self.cpg.switchGait("trot", self.gait.T_sw)
                self.motion_init = True

            legs_contact, legs_duty, if_switch_leg = self.cpg.update()

            X_ref_rt = self.gait.X_ref[self.ref_cnt]
            xi_ref_rt = self.gait.xi_ref[
                self.ref_cnt : (self.ref_cnt + self.mpc_param_.Nt)
            ]

            if closed_loop == 1:
                R_ = quat2rotm(self.state.qwxyz)
            else:
                R_ = np.eye(3)
            X_se3_ = np.block([[R_, self.state.pose], [0, 0, 0, 1]])

            # Stance legs, compute optimal force for stance leg
            # Swing leg, compute continuous bezier trajectory
            st_rs_ = []  # footend pose w.r.t. body
            sw_rs_ = []
            sw_idx = []
            sw_duty = 0
            for i in range(4):
                if legs_contact[i] == 1:
                    lp = np.array(
                        [self.state.legs[i].pose[0, 0], 0, -self.gait.stance_height]
                    )
                    st_rs_.append(self.bhip[i] + lp)
                    """ st_rs_.append(
                        self.bhip[i]
                        + np.array(
                            [
                                self.state.legs[i].pose[0, 0],
                                0,
                                self.state.legs[i].pose[1, 0],
                            ]
                        )
                    ) """
                else:
                    st_rs_.append(self.bhip[i] + np.array([0, 0, 0]))
                    sw_rs_.append(self.state.legs[i].pose.reshape(2))
                    sw_duty = legs_duty[i]
                    sw_idx.append(i)

            # st_u = self.mpc_controller.mpc(X_se3_, X_ref_rt, self.state.twist, xi_ref_rt, st_rs_)
            st_u = np.zeros(12)

            # Footend Position cmd generate by body pose, orientation tracking
            # translation
            self.body_pos_pid.setpoint = X_ref_rt[:3, 3].reshape(3)
            pid_comp = self.body_pos_pid.update(self.state.pose.reshape(3)).reshape(
                3, 1
            )
            if pid_comp[0, 0] > self.gait.vx * self.dt * 14:
                pid_comp[0, 0] = self.gait.vx * self.dt * 14
            elif pid_comp[0, 0] < -self.gait.vx * self.dt * 14:
                pid_comp[0, 0] = -self.gait.vx * self.dt * 14

            next_pose = self.state.pose + pid_comp
            # orientation
            desired_zyx = R.from_matrix(X_ref_rt[:3, :3]).as_euler("zyx")
            self.body_pitch_pid.setpoint = desired_zyx[1]
            current_zyx = R.from_matrix(R_).as_euler("zyx")
            current_pitch = current_zyx[1]
            d_pitch = self.body_pitch_pid.update(current_pitch)
            next_rmat = (
                R.from_euler("zyx", [0, d_pitch[0], 0]).as_matrix() @ X_se3_[:3, :3]
            )
            X_se3_next = np.block([[next_rmat, next_pose], [0, 0, 0, 1]])

            st_rs_next = []
            sw_rs_next = []

            for i in range(4):
                if legs_contact[i] == 1:
                    if self.close_loop == 1:
                        st_rs_next.append(
                            (
                                np.linalg.inv(X_se3_next)
                                @ X_se3_
                                @ np.block([[st_rs_[i].reshape(3, 1)], [1]])
                            )[:3, 0]
                            - self.bhip[i]
                        )
                    else:
                        # openloop mode
                        r_ = (
                            np.linalg.inv(X_ref_rt)
                            @ self.X_ref_rt_prev
                            @ np.block([[self.st_rs_prev[i].reshape(3, 1)], [1]])
                        )[:3, 0]
                        st_rs_next.append(r_)
                        self.st_rs_prev[i] = r_

            # Swing Legs
            if if_switch_leg or self.trot_swing_profiles == None:
                print("cnt, ", self.ref_cnt, "sw_idx, ", sw_idx)
                self.trot_swing_profiles = []
                for i, sidx in enumerate(sw_idx):
                    vx_ = self.gait.vx
                    vx1_ = self.gait.vx

                    if closed_loop == 1:
                        w_liftoff_pt = (
                            self.state.pose.reshape(3)
                            + self.bhip[sidx]
                            + np.array(
                                [
                                    self.state.legs[sidx].pose[0, 0],
                                    0,
                                    self.state.legs[sidx].pose[1, 0],
                                ]
                            )
                        )
                    else:
                        w_liftoff_pt = (
                            self.X_ref_rt_prev[:3, 3].reshape(3)
                            + self.bhip[sidx]
                            + np.array(
                                [
                                    self.state.legs[sidx].pose[0, 0],
                                    0,
                                    self.state.legs[sidx].pose[1, 0],
                                ]
                            )
                        )

                    self.swing_landpt[sidx] = self.swing_landpt[sidx] + np.array(
                        [self.gait.step_length, 0, 0]
                    )
                    w_touchdown_pt = self.swing_landpt[sidx]

                    w_liftoff_pt = np.array([w_liftoff_pt[0], w_liftoff_pt[2]])
                    w_touchdown_pt = np.array([w_touchdown_pt[0], w_touchdown_pt[2]])

                    # w_liftoff_pt[1] = 0.007
                    # w_touchdown_pt[1] = 0.007
                    w_liftoff_pt[1] = 0
                    w_touchdown_pt[1] = 0

                    if self.trot_swing_profiles != None:
                        st_ft = self.st_rs_fb_prev[sidx] - self.bhip[sidx]
                        st_ft = np.array([st_ft[0], st_ft[2]])
                        sw_ft = sw_rs_[i]
                        d_ft = sw_ft - st_ft
                        v_ = d_ft / self.dt
                        if v_[0] < -0.005:
                            vx_ = v_[0]
                        else:
                            vx_ = -0.005

                    print("swing_leg_state, ", self.state.legs[sidx].pose.reshape(2))
                    print("w_liftoff_pt, ", w_liftoff_pt)
                    print("w_touchdown_pt, ", w_touchdown_pt)
                    print("vx_, ", vx_)
                    print("-")

                    sp_ = self.swp.solveSwingTrajectory(
                        w_liftoff_pt,
                        w_touchdown_pt,
                        self.gait.step_height,
                        np.array([-vx_, 0]),
                        np.array([-vx1_, 0]),
                    )
                    self.trot_swing_profiles.append(sp_)
                print("----")

            for i, sidx in enumerate(sw_idx):
                w_swing_pt = self.trot_swing_profiles[i].getFootendPoint(sw_duty)
                b_swing_pt = w_swing_pt - (
                    np.array(np.array([X_ref_rt[0, 3], X_ref_rt[2, 3]]))
                    + np.array([self.bhip[sidx][0], self.bhip[sidx][2]])
                )
                # b_swing_pt = w_swing_pt - (np.array(np.array([self.state.pose[0,0], self.state.pose[2,0]])) + np.array([self.bhip[sw_idx][0], self.bhip[sw_idx][2]]))

                sw_rs_next.append(np.array([[b_swing_pt[0]], [0], [b_swing_pt[1]]]))
                if self.close_loop == 0:
                    self.st_rs_prev[sidx] = np.array(
                        [[b_swing_pt[0]], [0], [b_swing_pt[1]]]
                    )

            # Combine stance and swing leg command
            st_idx_ = 0
            sw_idx_ = 0
            rs_next = []
            for i in range(4):
                if legs_contact[i] == 1:
                    rs_next.append(st_rs_next[st_idx_])
                    st_idx_ += 1
                else:
                    rs_next.append(sw_rs_next[sw_idx_])
                    sw_idx_ += 1

            self.commandPublish(st_u, rs_next, self.imp_param, X_ref_rt, xi_ref_rt[0])

            if closed_loop == 1:
                self.logger.update_data(
                    self.seq,
                    X_se3_[0, 3],
                    X_se3_[2, 3],
                    current_pitch,
                    X_ref_rt[0, 3],
                    X_ref_rt[2, 3],
                    desired_zyx[1],
                    self.state.twist[3, 0],
                    self.state.twist[5, 0],
                    self.state.twist[2, 0],
                    self.state.x_raw,
                )
            else:
                # r_ = quat2rotm(self.state.filtered_qwxyz)
                r_ = quat2rotm(self.state.qwxyz)
                p_ = R.from_matrix(r_).as_euler("zyx")[1]
                self.logger.update_data(
                    self.seq,
                    self.state.pose[0, 0],
                    self.state.pose[2, 0],
                    p_,
                    X_ref_rt[0, 3],
                    X_ref_rt[2, 3],
                    desired_zyx[1],
                    self.state.twist[3, 0],
                    self.state.twist[5, 0],
                    self.state.twist[2, 0],
                    self.state.x_raw,
                )

            if self.ref_cnt < len(self.gait.X_ref) - self.mpc_param_.Nt - 1:
                self.ref_cnt += 1
                self.X_ref_rt_prev = X_ref_rt
                self.st_rs_fb_prev = st_rs_
            elif not self.if_track_finished:
                print("tracking complete, ", self.ref_cnt)
                self.if_track_finished = True
            else:
                self.logger.savefig()
                self.logger.savedata()
                self.fsm = "idle"

    def commandPublish(self, u, rs, imp_param, X_ref=None, xi_ref=None, seq=0):
        robot_msg = RobotStamped()
        robot_msg.header.seq = seq
        # robot_msg.stamp = rospy.Time.now()
        robot_msg.msg_type = "force"

        robot_msg.A_LF.force.pose_x = rs[0][0]
        robot_msg.A_LF.force.pose_y = rs[0][2]
        robot_msg.A_LF.force.force_x = u[0]
        robot_msg.A_LF.force.force_y = u[2]

        # robot_msg.A_LF.impedance.M_x = imp_param.M[0]
        # robot_msg.A_LF.impedance.M_y = imp_param.M[1]
        # robot_msg.A_LF.impedance.K0_x = imp_param.K0[0]
        # robot_msg.A_LF.impedance.K0_y = imp_param.K0[1]
        # robot_msg.A_LF.impedance.D_x = imp_param.D[0]
        # robot_msg.A_LF.impedance.D_y = imp_param.D[1]
        # robot_msg.A_LF.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        # robot_msg.A_LF.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        # robot_msg.A_LF.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        # robot_msg.A_LF.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        # robot_msg.A_LF.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        # robot_msg.A_LF.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.B_RF.force.pose_x = rs[1][0]
        robot_msg.B_RF.force.pose_y = rs[1][2]
        robot_msg.B_RF.force.force_x = u[3]
        robot_msg.B_RF.force.force_y = u[5]
        # robot_msg.B_RF.impedance.M_x = imp_param.M[0]
        # robot_msg.B_RF.impedance.M_y = imp_param.M[1]
        # robot_msg.B_RF.impedance.K0_x = imp_param.K0[0]
        # robot_msg.B_RF.impedance.K0_y = imp_param.K0[1]
        # robot_msg.B_RF.impedance.D_x = imp_param.D[0]
        # robot_msg.B_RF.impedance.D_y = imp_param.D[1]
        # robot_msg.B_RF.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        # robot_msg.B_RF.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        # robot_msg.B_RF.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        # robot_msg.B_RF.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        # robot_msg.B_RF.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        # robot_msg.B_RF.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.C_RH.force.pose_x = rs[2][0]
        robot_msg.C_RH.force.pose_y = rs[2][2]
        robot_msg.C_RH.force.force_x = u[6]
        robot_msg.C_RH.force.force_y = u[8]
        # robot_msg.C_RH.impedance.M_x = imp_param.M[0]
        # robot_msg.C_RH.impedance.M_y = imp_param.M[1]
        # robot_msg.C_RH.impedance.K0_x = imp_param.K0[0]
        # robot_msg.C_RH.impedance.K0_y = imp_param.K0[1]
        # robot_msg.C_RH.impedance.D_x = imp_param.D[0]
        # robot_msg.C_RH.impedance.D_y = imp_param.D[1]
        # robot_msg.C_RH.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        # robot_msg.C_RH.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        # robot_msg.C_RH.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        # robot_msg.C_RH.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        # robot_msg.C_RH.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        # robot_msg.C_RH.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

        robot_msg.D_LH.force.pose_x = rs[3][0]
        robot_msg.D_LH.force.pose_y = rs[3][2]
        robot_msg.D_LH.force.force_x = u[9]
        robot_msg.D_LH.force.force_y = u[11]
        # robot_msg.D_LH.impedance.M_x = imp_param.M[0]
        # robot_msg.D_LH.impedance.M_y = imp_param.M[1]
        # robot_msg.D_LH.impedance.K0_x = imp_param.K0[0]
        # robot_msg.D_LH.impedance.K0_y = imp_param.K0[1]
        # robot_msg.D_LH.impedance.D_x = imp_param.D[0]
        # robot_msg.D_LH.impedance.D_y = imp_param.D[1]
        # robot_msg.D_LH.impedance.adaptive_kp_x = imp_param.K_pid_x[0]
        # robot_msg.D_LH.impedance.adaptive_ki_x = imp_param.K_pid_x[1]
        # robot_msg.D_LH.impedance.adaptive_kd_x = imp_param.K_pid_x[2]
        # robot_msg.D_LH.impedance.adaptive_kp_y = imp_param.K_pid_y[0]
        # robot_msg.D_LH.impedance.adaptive_ki_y = imp_param.K_pid_y[1]
        # robot_msg.D_LH.impedance.adaptive_kd_y = imp_param.K_pid_y[2]

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
    with open(filepath, "r") as file:
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
    global filter_mode
    global com_shift
    global accel_L
    closed_loop = data["closed_loop"]
    standup_duration = data["standup_duration"]
    standup_idle_duration = data["standup_idle_duration"]
    step_length = data["step_length"]
    step_height = data["step_height"]
    stance_height = data["stance_height"]
    com_shift = data["com_shift"]
    T_sw = data["T_sw"]
    cycle = data["cycle"]
    gait_tolx_1 = data["gait_tolx_1"]
    gait_tolx_2 = data["gait_tolx_2"]
    pose_x_pid = data["pose_x_pid"]
    pose_z_pid = data["pose_z_pid"]
    orient_p_pid = data["orient_pitch_pid"]
    filter_x_freq = data["filter_x_freq"]
    filter_z_freq = data["filter_z_freq"]
    filter_pitch_freq = data["filter_pitch_freq"]
    accel_L = data["accel_L"]
    noise_mode = data["noise_mode"]
    filter_mode = data["filter_mode"]


def input_trigger(cobj):
    while True:
        mode = str(input("Mode Command [i]dle [t]rot [s]tand: "))
        if mode == "i":
            cobj.fsm = "idle"
        elif mode == "t":
            cobj.fsm = "trot"
        elif mode == "s":
            cobj.fsm = "standup"
        time.sleep(0.5)


if __name__ == "__main__":
    print("MPC started")
    rospy.init_node("corgi_mpc", anonymous=True)

    loadConfig(
        filepath="/home/biorola/corgi_ros_ws/src/corgi_mpc/config/config_trot.yaml"
    )
    # loadConfig(filepath="/root/corgi_rosws/src/corgi_mpc/config/config.yaml")

    cmpc = Corgi_mpc()
    cmpc.initialize()

    # terminal ui
    thread = threading.Thread(target=input_trigger, args=(cmpc,))
    thread.start()

    # cmpc.fsm = "standup"

    """ # plotter = RealTimePlot()
    # while True:
    #     plotter.update_plot(cmpc.state.raw_pose[0,0], cmpc.state.filtered_pose[0,0], cmpc.state.raw_pose[2,0], cmpc.state.filtered_pose[2,0])
    #     time.sleep(0.01) """

    rospy.spin()
