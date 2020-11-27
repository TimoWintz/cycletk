from cycletk import zwoparse
import cycletk
import pandas as pd
import numpy as np
from cycletk.activities import Activity

def read(filename):
    """Read a zwo workout file .

    Args:
        filename (str): target file name

    Returns:
        Workout: workout object
    """    
    x = open(filename, "r").read()
    wo_dict = zwoparse.parse(x)
    df = Workout(index = range(len(wo_dict["segments"])))
    df.name = wo_dict["name"]
    df.description = wo_dict["description"]
    df["type"] = [s.segment_type for s in wo_dict["segments"]]
    df["start_time"] = [pd.Timedelta(seconds=s.start_time) for s in wo_dict["segments"]]
    df["end_time"] = [pd.Timedelta(seconds=s.end_time) for s in wo_dict["segments"]]
    df["intensity_start"] = np.array([s.power.min_intensity for s in wo_dict["segments"]], dtype=np.float32)
    df["intensity_end"] = np.array([s.power.max_intensity for s in wo_dict["segments"]], dtype=np.float32)
    return df

class Workout(pd.DataFrame):
     # normal properties
    _metadata = ['name', 'description']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, rider):
        x,y = self.curve(rider)
        pd.Series(y, index=x).plot()

    def normalized_power(self, rider):
        avg = 0
        for wo_index in self.index:
            st = self["start_time"][wo_index]
            et = self["end_time"][wo_index]
            pmax = self["intensity_end"][wo_index] * rider.cp
            pmin = self["intensity_start"][wo_index] * rider.cp
            if (pmax < pmin):
                pmax, pmin = pmin, pmax
            if self["type"][wo_index] == "warmup" or self["type"][wo_index] == "cooldown":
                avg += (pmax**4 + pmax**3*pmin + pmax**2 * pmin**2 +
                            pmax * pmin**3 + pmin**4)*(et - st).total_seconds() / 5
            else:
                avg += (et - st).total_seconds() * pmax ** 4
        avg /=  (et - self["start_time"][0]).total_seconds()
        avg = avg ** (1/4)
        return avg

    def intensity(self, rider):
        return self.normalized_power(rider) / rider.cp

    def stress(self, rider):
        return 100 * self.intensity(rider) * (self["end_time"][self.index[-1]]
            - self["start_time"][self.index[0]]).total_seconds() / 3600

    def curve(self, rider):
        from matplotlib import pyplot as plt 

        x = []
        y = []
        for wo_index in self.index:
            st = self["start_time"][wo_index].total_seconds()
            et = self["end_time"][wo_index].total_seconds()
            x.extend([st, et])
            if self["type"][wo_index] == "warmup" or self["type"][wo_index] == "cooldown":
                y.extend([self["intensity_start"][wo_index] * rider.cp, self["intensity_end"][wo_index] * rider.cp])
            else:
                y.extend([self["intensity_end"][wo_index]* rider.cp, self["intensity_end"][wo_index]* rider.cp])
        return x,y

    def to_activity(self, rider, dt_s=1):
        index = np.arange(self["start_time"].min(), self["end_time"].max(), dt_s*1000*1000)
        activity = Activity(index=index)
        pwr = np.zeros(len(index))
        for wo_index in self.index:
            st = self["start_time"][wo_index]
            et = self["end_time"][wo_index]
            current_index_mask = np.logical_and(index >= st, index < et)
            current_index = index[current_index_mask]
        
            if self["type"][wo_index] == "warmup" or self["type"][wo_index] == "cooldown":
                pwr[current_index_mask] = (current_index - st) / (et - st) * self["intensity_start"][wo_index] * rider.cp + (et - current_index) / (et - st) * self["intensity_end"][wo_index] * rider.cp
            else:
                pwr[current_index_mask] = self["intensity_end"][wo_index] * rider.cp
        activity["pwr"] = pwr
        return activity
