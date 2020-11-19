from cycletk.activity import power_curve
from sklearn.linear_model import LinearRegression
import numpy as np

# Constants for power curve
DT_S = 15
TMAX_S = 3600
TMAX_S_EST = 1200 # for estimation
TMIN_S_EST = 60 # for estimation


class Rider:
    PZ_NAMES = ["Recovery", "Endurance", "Tempo", "Threshold", "VO2Max", "Anaerobic", "Neuromuscular"]

    def __init__(self, cp, wp, weight):
        self.cp = cp
        self.wp = wp
        self.weight = weight
        self.power_curve = None
        self.pz_bins = np.array([0, 0.55, 0.75, 0.9, 1.05, 1.2, 1.5])
        

    def update_power_curve(self, activity):
        if self.power_curve is None:
            self.power_curve = power_curve(activity, DT_S, TMAX_S)
        else:
            new_power_curve = power_curve(activity, DT_S, TMAX_S)
            self.power_curve["new_max_pwr"] = new_power_curve["max_pwr"]
            self.power_curve["max_pwr"] = np.maximum(self.power_curve["max_pwr"], self.power_curve["new_max_pwr"])
                

    def estimate_model(self):
        if self.power_curve is None:
            return
        t_inv = np.asarray(1 / self.power_curve.index.total_seconds())

        idx = np.logical_and((t_inv > 1/TMAX_S_EST),(t_inv < 1/TMIN_S_EST))
        t_inv = t_inv[idx].reshape(-1,1)
        
        reg = LinearRegression().fit(t_inv, self.power_curve["max_pwr"][idx])
        self.wp = reg.coef_[0]
        self.cp = reg.intercept_
        self.power_curve["est_max_pwr"] = self.cp + self.wp / self.power_curve.index.total_seconds()
