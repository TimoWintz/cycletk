import numpy as np
from tqdm import tqdm
import pandas as pd
import activityio

def normalized_power(activity, dt_s=1):
    return ((activity["pwr"].rolling(pd.Timedelta(seconds=dt_s)).mean()**4).mean())**(1/4)

def average_power(activity, dt_s=1):
    return activity["pwr"].rolling(pd.Timedelta(seconds=dt_s)).mean().mean()

def maximum_power(activity, dt_s=1):
    return activity["pwr"].rolling(pd.Timedelta(seconds=dt_s)).min().max()

def set_wp_bal(activity, rider): # Formula from Skiba 2015
    dt_s = (activity.index[1:] - activity.index[:-1]).total_seconds()
    
    power = activity["pwr"]
    dp_pos = np.maximum(power - rider.cp, 0)
    dp_neg = np.minimum(power - rider.cp, 0)
    rec_factor = dp_neg / rider.wp # skiba 2015

    n_idx = len(activity.index)
    wp_exp = np.zeros(n_idx)
    for i in range(1, n_idx):
        wp_exp[i] = (wp_exp[i-1] + dt_s[i-1] * dp_pos[i]) * np.exp(rec_factor[i] * dt_s[i-1])
    activity["wp_bal"] = rider.wp - wp_exp
    return activity

def min_wp_bal(activity, rider): # Formula from Skiba 2015
    new_activity = activity.copy()
    set_wp_bal(new_activity, rider)
    return new_activity["wp_bal"].min()

def intensity(activity, rider):
    return normalized_power(activity) / rider.cp

def stress(activity, rider):
    return 100 * intensity(activity, rider) * (activity.index[-1] - activity.index[0]).total_seconds() / 3600

def power_curve(activity, dt_s=30, tmax_s=3600):
    index = np.arange(dt_s, tmax_s, dt_s)
    index = [pd.Timedelta(seconds=i) for i in index]
    x = pd.DataFrame(index=index)
    x["max_pwr"] = 0.0
    for i in index:
        if (i < activity.index[-1] - activity.index[0]):
            x["max_pwr"][i] = activity["pwr"].rolling(i).mean().max()
    return x

def power_zones(activity, rider):
    activity["pz"] = 0
    for i in range(len(rider.pz_bins)):
        idx = np.copy(activity["pwr"] >= rider.pz_bins[i] * rider.cp)
        activity.loc[idx, "pz"] = i
    df = activity.copy()
    df.resample(pd.Timedelta(seconds=1))
    df['ones'] = 1
    df = df.groupby("pz").sum()
    df['time'] = df['ones'].apply(lambda x: pd.Timedelta(seconds=x))
    df['percent'] = df['time'] / df['time'].sum() * 100
    df.index = rider.PZ_NAMES
    return df[['time', 'percent']]  