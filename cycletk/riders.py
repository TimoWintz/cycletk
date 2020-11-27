from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Constants for power curve
DT_S = 15
TMAX_S = 3600
TMAX_S_EST = 1200 # for estimation
TMIN_S_EST = 60 # for estimation


class Rider:
    PZ_NAMES = ["Recovery", "Endurance", "Tempo", "Threshold", "VO2Max", "Anaerobic", "Neuromuscular"]

    def __init__(self, cp, wp, weight):
        """Initialize Rider.

        Args:
            cp (float): Critical Power
            wp (float): Anaerobic reserve
            weight (float): Rider weight
        """        
        self.cp = cp
        self.wp = wp
        self.weight = weight
        self._power_curve = None
        self.pz_bins = np.array([0, 0.55, 0.75, 0.9, 1.05, 1.2, 1.5])
        

    def add_power_curve(self, activity):
        """Update the power curve .

        Args:
            activity (Activity): activity to add.
        """        
        if self._power_curve is None:
            self._power_curve = activity.power_curve(DT_S, TMAX_S).rename(columns={"pwr", str(activity.start)})
        else:
            new_power_curve = activity.power_curve(DT_S, TMAX_S)
            self._power_curve[str(activity.start)] = new_power_curve["max_pwr"]

    def power_curve(self, period=None):
        """Returns the power curve .

        Args:
            period (tuple, optional): Tuple of pd.Timestamp. Defaults to None.

        Returns:
            pd.Dataframe : power curve
        """        
        period_begin, period_end = period
        columns = [c for c in self._power_curve.columns() if pd.Timestamp(c) <= period_end and pd.Timestamp(c) > period_begin]
        return self._power_curve[columns].max(axis=1)
                
    def estimate_model(self, period=None):
        """Estimate the Wpbal model parameters from power curve
        """        
        if self._power_curve is None:
            return
        t_inv = np.asarray(1 / self._power_curve.index.total_seconds())

        idx = np.logical_and((t_inv > 1/TMAX_S_EST),(t_inv < 1/TMIN_S_EST))
        t_inv = t_inv[idx].reshape(-1,1)
        
        reg = LinearRegression().fit(t_inv, self.power_curve(period)[idx])
        cp = reg.coef_[0]
        wp = reg.intercept_
        return cp, wp
        

class RiderSetup:
    def __init__(self, total_mass, drivetrain_efficiency=0.97, cda=0.035, cr=0.003):
        self.total_mass = total_mass
        self.drivetrain_efficiency = drivetrain_efficiency
        self.cda = cda
        self.cr = cr
