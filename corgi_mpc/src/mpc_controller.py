#!/usr/bin/env python3
import rospy
from corgi_ros_bridge.msg import RobotStamped
import numpy as np

from quad_mpc_3df import *
import matplotlib.pyplot as plt


class ImpedanceParam:
    def __init__(self):
        self.M = np.array([0.652, 0.652])  # Mx, My
        self.K0 = np.array([30000, 30000])  # Kx, Ky
        # self.K0 = np.array([1e6, 1e6])  # Kx, Ky
        self.D = np.array([400, 400])  # Dx, Dy
        self.K_pid_x = np.array([2000, 1800, 50])
        self.K_pid_y = np.array([2000, 1800, 50])
        # self.K_pid_x = np.array([0, 0, 0])
        # self.K_pid_y = np.array([0, 0, 0])


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


class Corgi_mpc:
    def __init__(self, dt=0.01):
        self.timer = rospy.Timer(rospy.Duration(dt), self.loop)

        self.state = RobotState()
        self.state_sub = rospy.Subscriber("/robot/state", RobotStamped, self.stateCallback)
        self.force_pub = rospy.Publisher("/robot/command", RobotStamped, queue_size=1024)

        self.dt = dt

        self.mpc_param_ = None
        self.mpc_controller = None
        self.pos_param = ImpedanceParam()
        self.pos_param.K0 = np.array([1e6, 1e6])
        self.pos_param.K_pid_x = np.zeros(3)
        self.pos_param.K_pid_y = np.zeros(3)
        self.imp_param = ImpedanceParam()

        self.fsm = "idle"
        self.standup_time = 2
        self.stand_fttraj = []
        self.stand_cnt = 0

        self.X_ref_rt_prev = None
        self.rs_prev = None
        w = 329.5 * 0.001
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

        self.body_pos_pid = VectorPIDController([1, 0, 1], [0.8, 0, 0.8], [0.02, 0, 0.02], dt, setpoint=[0, 0, 0])
        self.body_pitch_pid = VectorPIDController(
            np.array([1]),
            np.array([0.9]),
            np.array([0.0]),
            dt,
            setpoint=np.array([0]),
        )
        self.logger = logger()

    def initialize(self):
        self.reference_conf()
        self.mpc_controller = MPC(self.mpc_param_)
        self.if_init = True

    def reference_conf(self):
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

        # Generate Reference trajectory
        p0_ref = np.array([[0.0], [0], [0.21]])
        q0_wxyz_ref = np.array([[1], [0], [0], [0]])
        R0_ref = quat2rotm(q0_wxyz_ref)

        w0_ref = np.array([[0], [-0.065], [0]])
        v0_ref = np.array([[0.02], [0], [0]])
        X0_ref = np.block([[R0_ref, p0_ref], [np.zeros([1, 3]), 1]])
        xid_ref = np.block([[w0_ref], [v0_ref]])

        X_refs = [X0_ref.copy() for _ in range(Nsim)]
        xi_refs = [xid_ref.copy() for _ in range(Nsim)]
        X = X0_ref

        for i in range(1, Nsim):
            xid_ref_rt = xid_ref
            Xi_ = np.block([[skew(xid_ref_rt[:3]), xid_ref_rt[3:]], [0, 0, 0, 0]])
            X = X @ sp.linalg.expm(Xi_ * dt)
            X_refs[i] = X
            xi_refs[i] = xid_ref_rt.copy()
        self.mpc_param_.X_ref = X_refs  # in SE3 form
        self.mpc_param_.xi_ref = xi_refs  # twist 6x1

        self.X_ref_rt_prev = X_refs[0]

        self.stand_fttraj = np.linspace(-0.1, -p0_ref[2, 0], int(3 / self.dt))

    def stateCallback(self, data):
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

    def loop(self, event):
        print("[Mode] ", self.fsm)

        if self.fsm == "idle":
            pass

        elif self.fsm == "standup":
            if np.abs(self.state.pose[2] - self.mpc_param_.X_ref[0][2, 3]) < 0.01:
                self.fsm = "mpc"
                print("switch to mpc")
            else:
                rs_ = [np.array([0, 0, self.stand_fttraj[self.stand_cnt]]) for i in range(4)]
                if self.stand_cnt < len(self.stand_fttraj) - 1:
                    self.stand_cnt += 1
                print(rs_[0])
                u_ = np.ones([12, 1]) * 0
                self.rs_prev = rs_
                self.commandPublish(u_, rs_, self.pos_param)

        elif self.fsm == "mpc":
            X_ref_rt = self.mpc_param_.X_ref[self.ref_cnt]
            xi_ref_rt = self.mpc_param_.xi_ref[self.ref_cnt : (self.ref_cnt + self.mpc_param_.Nt)]

            X_se3_ = np.block([[quat2rotm(self.state.qwxyz), self.state.pose], [0, 0, 0, 1]])

            rs_ = []
            for i in range(4):
                rs_.append(
                    self.bhip[i]
                    + np.array(
                        [
                            self.state.legs[i].pose[0, 0],
                            0,
                            self.state.legs[i].pose[1, 0],
                        ]
                    )
                )

            start = time.time()
            u = self.mpc_controller.mpc(X_se3_, X_ref_rt, self.state.twist, xi_ref_rt, rs_)
            end = time.time()
            print("time_elapsed, ", end - start)

            """ # calculate total wrench
            b0_ = np.block([[skew(rs_[0])], [np.eye(3)]])
            b1_ = np.block([[skew(rs_[1])], [np.eye(3)]])
            b2_ = np.block([[skew(rs_[2])], [np.eye(3)]])
            b3_ = np.block([[skew(rs_[3])], [np.eye(3)]])
            b_ = np.hstack([b0_, b1_, b2_, b3_])
            u_total = b_ @ u.reshape(12, 1) """

            # translation tracking
            self.body_pos_pid.setpoint = X_ref_rt[:3, 3].reshape(3)
            next_pose = self.state.pose + self.body_pos_pid.update(self.state.pose.reshape(3)).reshape(3, 1)

            # orientation tracking
            desired_zyx = R.from_matrix(X_ref_rt[:3, :3]).as_euler("zyx")
            self.body_pitch_pid.setpoint = desired_zyx[1]
            current_zyx = R.from_quat(
                np.array(
                    [
                        self.state.qwxyz[1],
                        self.state.qwxyz[2],
                        self.state.qwxyz[3],
                        self.state.qwxyz[0],
                    ]
                ).reshape(4)
            ).as_euler("zyx")
            current_pitch = current_zyx[1]
            d_pitch = self.body_pitch_pid.update(current_pitch)
            next_rmat = R.from_euler("zyx", [0, d_pitch[0], 0]).as_matrix() @ X_se3_[:3, :3]
            X_se3_next = np.block([[next_rmat, next_pose], [0, 0, 0, 1]])

            print("--")
            print("d_pitch, ", d_pitch)
            print("current_pitch, ", current_pitch)
            print(X_se3_)
            print(X_se3_next)
            print("rs0")
            print(rs_)

            rs_next = []

            """ for rs in self.rs_prev:
                rs_next.append(
                    (
                        np.linalg.inv(X_se3_next)
                        @ self.X_ref_rt_prev
                        @ np.block([[rs.reshape(3, 1)], [1]])
                    )[:3, 0]
                ) """

            """ for rs in self.rs_prev:
                rs_next.append(
                    (
                        np.linalg.inv(X_se3_next)
                        @ X_se3_
                        @ np.block([[rs.reshape(3, 1)], [1]])
                    )[:3, 0]
                )
                print(
                    (
                        np.linalg.inv(X_se3_next)
                        @ X_se3_
                        @ np.block([[rs.reshape(3, 1)], [1]])
                    )[:3, 0]
                ) """

            for i in range(4):
                rs_next.append(
                    (np.linalg.inv(X_se3_next) @ X_se3_ @ np.block([[rs_[i].reshape(3, 1)], [1]]))[:3, 0]
                    - self.bhip[i]
                )
                print((np.linalg.inv(X_se3_next) @ X_se3_ @ np.block([[rs_[i].reshape(3, 1)], [1]]))[:3, 0])

            self.commandPublish(u, rs_next, self.imp_param, X_ref_rt, xi_ref_rt[0])

            self.logger.update_data(
                X_se3_[0, 3],
                X_se3_[2, 3],
                current_pitch,
                X_ref_rt[0, 3],
                X_ref_rt[2, 3],
                desired_zyx[1],
            )

            if self.ref_cnt < len(self.mpc_param_.X_ref) - self.mpc_param_.Nt - 1:
                self.ref_cnt += 1
                self.X_ref_rt_prev = X_ref_rt
                self.rs_prev = rs_next
            elif not self.if_track_finished:
                print("tracking complete")
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
        # print("cmd published")


if __name__ == "__main__":
    print("MPC started")
    rospy.init_node("corgi_mpc", anonymous=True)

    cmpc = Corgi_mpc()
    cmpc.initialize()
    cmpc.fsm = "standup"
    # cmpc.whileloop()
    rospy.spin()
