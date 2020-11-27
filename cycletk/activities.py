import numpy as np
from tqdm import tqdm
import pandas as pd
import activityio
from activityio._types import ActivityData
from opendata import OpenData
import os


def read(filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext in [".fit", ".tcx", ".gpx", ".pwx", ".srm"]:
        activity = activityio.read(filename)
    elif ext == ".json":
        activity = pd.read_json(filename)
    df = Activity(activity)
    df.start = activity.start

    df.source_type = "file"
    df.source = os.path.abspath(filename)
    return df

def read_golden_cheetah(athlete_id, activity_id, target_file=None):
    od = OpenData()
    athlete = od.get_local_athlete(athlete_id)
    gc_activity = athlete.get_activity(activity_id)
    

    if gc_activity.has_data():
        # pylint: disable=no-member, unsubscriptable-object
        activity  = Activity(index=[pd.Timedelta(seconds=s) for s in gc_activity.data.secs])
        
        
        activity.resample(pd.Timedelta(seconds=1))
        if not np.isnan(np.array(gc_activity.data["km"])).all():
            activity["dist"] = np.array(gc_activity.data["km"], dtype=np.float) * 1000
            activity["speed"] = np.gradient(activity["dist"])
        if not np.isnan(np.array(gc_activity.data["power"])).all():
            activity["pwr"] = np.array(gc_activity.data["power"], dtype=np.float)
        for key in ["alt", "cad", "hr"]:
            if not np.isnan(np.array(gc_activity.data[key])).all():
                activity[key] = np.array(gc_activity.data[key], dtype=np.float)
        
    else:
        activity  = Activity()

    activity.source_type = "golden_cheetah"   
    activity.start = pd.Timestamp(gc_activity.metadata["date"])
    activity.source = athlete_id + "," + activity_id

    return activity

class Activity(pd.DataFrame):
    def normalized_power(self, dt_s=1.0):
        """Return the activity normalized power

        Args:
            dt_s (float, optional): time step for discretization in s. Defaults to 1.

        Returns:
            float: NP in W
        """
        if "pwr" not in self.keys():
            return np.nan
        return ((self["pwr"].rolling(pd.Timedelta(seconds=dt_s)).mean()**4).mean())**(1/4)


    def average_power(self, dt_s=1.0):
        """Returns the average power .

        Args:
            dt_s (float, optional): time step for discretization in s. Defaults to 1.

        Returns:
            float: average power in W
        """
        if "pwr" not in self.keys():
            return np.nan  
        return self["pwr"].rolling(pd.Timedelta(seconds=dt_s)).mean().mean()


    def maximum_power(self, dt_s=1.0):
        """Returns the maximum power .

        Args:
            dt_s (float, optional): time step for discretization in s. Defaults to 1.

        Returns:
            float: maximum power in W
        """
        if "pwr" not in self.keys():
            return np.nan   
        return self["pwr"].rolling(pd.Timedelta(seconds=dt_s)).min().max()


    def set_wp_bal(self, rider):  # Formula from Skiba 2015
        if "pwr" not in self.keys():
            return self
        dt_s = (self.index[1:] - self.index[:-1]).total_seconds()

        power = self["pwr"]
        dp_pos = np.maximum(power - rider.cp, 0)
        dp_neg = np.minimum(power - rider.cp, 0)
        rec_factor = dp_neg / rider.wp  # skiba 2015

        n_idx = len(self.index)
        wp_exp = np.zeros(n_idx)
        for i in range(1, n_idx):
            wp_exp[i] = (wp_exp[i-1] + dt_s[i-1] * dp_pos[i]) * \
                np.exp(rec_factor[i] * dt_s[i-1])
        self["wp_bal"] = rider.wp - wp_exp
        return self


    def max_expended_wp(self, rider):  # Formula from Skiba 2015
        if "pwr" not in self.keys():
            return 0
        new_activity = self.copy()
        new_activity.set_wp_bal(rider)
        return rider.wp - new_activity["wp_bal"].min()


    def intensity(self, rider):
        return self.normalized_power() / rider.cp


    def stress(self, rider):
        return (100 * self.intensity(rider) *
            (self.index[-1] - self.index[0]).total_seconds() / 3600)

    def duration(self):
        return self.index[-1]

    def distance(self):
        if "dist" not in self.keys():
            return 0
        return self["dist"][-1]


    def power_curve(self, dt_s=30, tmax_s=3600):
        """Returns the power curve

        Args:
            dt_s (int, optional): time step in seconds. Defaults to 30.
            tmax_s (int, optional): max time interval in seconds. Defaults to 3600.

        Returns:
            pd.DataFrame: power curve
        """
        index = np.arange(dt_s, tmax_s, dt_s)
        index = [pd.Timedelta(seconds=i) for i in index]
        x = pd.DataFrame(index=index)
        x["max_pwr"] = 0.0
        if "pwr" not in self.keys():
            return x
        for i in index:
            if (i < self.index[-1] - self.index[0]):
                x["max_pwr"][i] = self["pwr"].rolling(i).mean().max()
        return x


    def power_zones(self, rider):
        """Power zones for a ride

        Args:
            rider (Rider): target rider

        Returns:
            pd.DataFrame: indexed by power zone, time and percentage spent is each zone
        """        
        self["pz"] = 0
        df = pd.DataFrame(self).copy()

        for i in range(len(rider.pz_bins)):
            idx = np.copy(self["pwr"] >= rider.pz_bins[i] * rider.cp)
            df.loc[idx, "pz"] = i
        
        df.resample(pd.Timedelta(seconds=1))
        df['ones'] = 1
        df = df.groupby("pz").sum()
        df['time'] = df['ones'].apply(lambda x: pd.Timedelta(seconds=x))
        df['percent'] = df['time'] / df['time'].sum() * 100
        missing_idx = [i for i in range(len(rider.PZ_NAMES)) if i not in df.index]
        df = df.append(pd.DataFrame({'time' : 0, 'percent' : 0 }, index=missing_idx))
        print(df.index)
        df.index = rider.PZ_NAMES
        return df[['time', 'percent']]


    def scores(self, rider):
        """Returns a DataFrame with scores averaged over the power zones .

        Args:
            rider (Rider): target rider

        Returns:
            pd.DataFrame: scores
        """        
        pz = self.power_zones(rider)
        scores = pd.DataFrame(index=["threshold", "vo2max", "sprint"])
        scores["scores"] = 0
        scores["scores"]["threshold"] = pz["percent"][2:4].sum()
        scores["scores"]["vo2max"] = 10*pz["percent"][4:6].sum()
        scores["scores"]["sprint"] = 100*pz["percent"][6]
        return scores["scores"]

    def save(self, filename):
        self.source_type = "file"
        self.file_location = os.path.abspath(filename)
        self.to_json(filename)