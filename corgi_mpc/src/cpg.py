import numpy as np
import time
import matplotlib.pyplot as plt


class CPG:
    def __init__(self, dt):
        self.gait = "stand"
        # stand, walk, trot ...

        self.t_iter = 0
        self.dt = dt

        self.legs_stance_timer = np.array([0, 0, 0, 0], dtype=np.float64)
        self.legs_contact = np.array([1, 1, 1, 1])  # 0 for swing, 1 for stance

        self.T_sw = 0
        self.T_st = 0
        self.swing_timer = 0
        self.pattern_idx = 0
        self.pattern = np.array([0, 2, 1, 3])
        self.if_switch_leg = False

        self.timer_logs = []

        self.sw_duty = 0  # swing duty
        self.st_duty = 0  # stance duty

    def fsm(self):
        if self.gait == "stand":
            pass
        elif self.gait == "walk":
            self.if_switch_leg = False
            if self.swing_timer >= self.T_sw:
                self.legs_contact[self.pattern[self.pattern_idx]] = 1

                self.pattern_idx += 1
                self.pattern_idx = self.pattern_idx % 4
                self.swing_timer = 0

                # update contact list
                self.legs_stance_timer[self.pattern[self.pattern_idx]] = 0
                self.legs_contact[self.pattern[self.pattern_idx]] = 0

                self.if_switch_leg = True

                # print("[CPG] time elapsed, ", self.t_iter)
                # print("[CPG] switch swing leg, ", self.pattern[self.pattern_idx])
                # print("[CPG] contact list, ", self.legs_contact)
                # print("--")

            for pidx in self.pattern:
                if pidx != self.pattern[self.pattern_idx]:
                    self.legs_stance_timer[pidx] = self.legs_stance_timer[pidx] + self.dt

            # self.timer_logs.append(self.legs_stance_timer.copy())
            self.swing_timer += self.dt

        elif self.gait == "walk_safe":
            self.if_switch_leg = False
            # if self.swing_timer - self.T_sw > 5 * self.T_sw:
            if np.abs(self.swing_timer - self.T_sw) < 0.0001:
                self.legs_contact[self.pattern[self.pattern_idx]] = 1

            elif self.swing_timer >= self.T_sw:
                self.legs_contact[self.pattern[self.pattern_idx]] = 1

                self.pattern_idx += 1
                self.pattern_idx = self.pattern_idx % 4
                self.swing_timer = 0

                # update contact list
                self.legs_stance_timer[self.pattern[self.pattern_idx]] = 0
                self.legs_contact[self.pattern[self.pattern_idx]] = 0

                self.if_switch_leg = True

                print("[CPG] time elapsed, ", self.t_iter)
                print("[CPG] switch swing leg, ", self.pattern[self.pattern_idx])
                print("[CPG] contact list, ", self.legs_contact)
                print("--")

            for pidx in self.pattern:
                if pidx != self.pattern[self.pattern_idx]:
                    self.legs_stance_timer[pidx] = self.legs_stance_timer[pidx] + self.dt
            self.swing_timer += self.dt

        elif self.gait == "trot":
            self.if_switch_leg = False
            if self.swing_timer >= self.T_sw:

                self.legs_contact[self.pattern[self.pattern_idx]] = 1
                self.legs_contact[self.pattern[self.pattern_idx + 1]] = 1

                self.pattern_idx += 2
                self.pattern_idx = self.pattern_idx % 4
                self.swing_timer = 0

                self.legs_stance_timer[self.pattern[self.pattern_idx]] = 0
                self.legs_stance_timer[self.pattern[self.pattern_idx + 1]] = 0

                self.legs_contact[self.pattern[self.pattern_idx]] = 0
                self.legs_contact[self.pattern[self.pattern_idx + 1]] = 0

                self.if_switch_leg = True

                # print("[CPG] time elapsed, ", self.t_iter)
                # print(
                #     "[CPG] switch swing leg, ",
                #     self.pattern[self.pattern_idx],
                #     ", ",
                #     self.pattern[self.pattern_idx + 1],
                # )
                # print("[CPG] contact list, ", self.legs_contact)
                # print("--")

            for pidx in self.pattern:
                if pidx != self.pattern[self.pattern_idx] or pidx != self.pattern[self.pattern_idx + 1]:
                    self.legs_stance_timer[pidx] = self.legs_stance_timer[pidx] + self.dt

            self.swing_timer += self.dt

    def update(self):
        self.fsm()

        # print("[CPG] contact list, ", self.legs_contact)

        self.t_iter += self.dt

        if self.gait == "stand":
            legs_duty = np.array([0, 0, 0, 0])
        elif self.gait == "walk":
            self.sw_duty = self.swing_timer / self.T_sw
            self.st_duty = self.legs_stance_timer / self.T_st
            legs_duty = self.legs_contact * self.st_duty + -(self.legs_contact - 1) * self.sw_duty
        else:
            self.sw_duty = self.swing_timer / self.T_sw
            legs_duty = np.array([self.sw_duty, self.sw_duty, self.sw_duty, self.sw_duty])
        # print("[CPG] duty list, ", legs_duty)
        return [self.legs_contact, legs_duty, self.if_switch_leg]

    def switchGait(self, next_gait, T_sw):
        self.gait = next_gait
        if next_gait == "walk" or next_gait == "walk_safe":
            self.T_sw = T_sw
            self.T_st = 3 * T_sw
            self.legs_contact[self.pattern[self.pattern_idx]] = 0
        elif next_gait == "trot":
            self.T_sw = T_sw
            self.T_st = T_sw
            self.legs_contact[self.pattern[self.pattern_idx]] = 0
            self.legs_contact[self.pattern[self.pattern_idx + 1]] = 0


if __name__ == "__main__":
    print("CPG")
    cpg = CPG(0.01)

    t = 0
    while t < 2:
        cpg.update()
        time.sleep(0.01)
        t += 0.01

    cpg.switchGait("trot", 0.6)

    while t < 10:
        cpg.update()
        time.sleep(0.01)
        t += 0.01

    # logs = np.vstack(cpg.timer_logs)
    # print(logs.shape)
    # plt.plot(logs[:, 0])
    # plt.plot(logs[:, 1])
    # plt.plot(logs[:, 2])
    # plt.plot(logs[:, 3])
    # plt.show()
