from scipy.optimize import bisect
import numpy as np
import ruptures as rpt
import pandas as pd


VMAX_MPS = 1000
GRAVITY = 9.81


def pressure_from_altitude(alt):
    """Calculate pressure from altitude .

    Args:
        alt (float): altitude in meters

    Returns:
        float: pressure in hpa
    """    
    return 1013.25 * (1-0.0065*alt/288.15)**(5.255)


def air_density_pressure(temp, pressure_hpa):
    """Calculate air density from temperature and pressure .

    Args:
        temp (float): temperature in Celcius degrees
        pressure_hpa (float): pressure in hpa

    Returns:
        air_density: air density in kg/m3
    """    
    R_air = 287
    temp_K = temp + 273
    pressure_pa = 100*pressure_hpa
    air_density = pressure_pa / temp_K / R_air
    return air_density


def air_density(alt, temp):
    """Calculate the air density at a given altitude .

    Args:
        alt (float): altitude in meters
        temp (float): temperature in °C

    Returns:
        float: air density in kg/m3
    """    
    return air_density_pressure(temp, pressure_from_altitude(alt))


def power_from_speed(rider_setup, grade, speed, altitude=0.0, temp=20.0, headwind=0.0):
    """Calculate steady state output power from vehicle speed .

    Args:
        rider_setup (RiderSetup): target rider setup
        grade (float): slope grade in percent
        speed (float): rider speed in m/s
        altitude (float, optional): altitude in meters. Defaults to 0.
        temp (float, optional): temperature in °C. Defaults to 20.
        headwind (float, optional): headwind in m/s. Defaults to 0.0.

    Returns:
        float: output power in W
    """    
    mass, drivetrain_efficiency, cda, cr = (rider_setup.total_mass, rider_setup.drivetrain_efficiency,
                  rider_setup.cda, rider_setup.cr)
    rho = air_density(altitude, temp)
    return (1/drivetrain_efficiency * speed * (0.5 * rho * cda * np.sign(speed+headwind) *
                                           (speed + headwind) * (speed + headwind) + cr *
                                           mass * GRAVITY + mass * GRAVITY *
                                           np.sin(np.arctan(grade / 100))))


def speed_from_power(rider_setup, grade, power, alt=0, temp=20, headwind=0.0):
    """Calculate steady state vehicle speed from output power .

    Args:
        rider_setup (RiderSetup): target rider setup
        grade (float): slope grade in percent
        power (float): output power in W
        altitude (float, optional): altitude in meters. Defaults to 0.
        temp (float, optional): temperature in °C. Defaults to 20.
        headwind (float, optional): headwind in m/s. Defaults to 0.0.

    Returns:
        float: rider speed in m/s
    """    
    return bisect(lambda x: power - power_from_speed(rider_setup, grade, x,
                                                     alt, temp, headwind), 0, VMAX_MPS)


def estimate_power(activity, rider_setup, pen=5, temp=None):
    """Estimate the power output for a ride .

    Args:
        activity (Activity): target activity
        rider_setup (RiderSetup): target rider setup
        pen (int, optional): regularization penalty. Defaults to 5.
        temp (float, optional): temperature in °C. By default, temperature will be taken from activity data.
    """    
    activity.resample(pd.Timedelta(seconds=1))
    values = np.array(activity["speed"].rolling(pd.Timedelta(seconds=3)).mean(), dtype=np.float64)
    values = np.gradient(values)
    values[np.isnan(values)] = 0.0

    activity["acceleration"] = values
    
    if temp is None:
        temp = 20
    algo = rpt.Pelt(model="rbf").fit(values)
    pen = pen*np.log(len(values))*np.std(values)**2
    indices = algo.predict(pen=pen)
    print("n = {0}".format(len(indices)))
    key_est = "pwr"
    activity[key_est] = 0
    
    for i in range(1, len(indices) - 1):
        altitude = activity["alt"][indices[i-1]]
        dist = activity["dist"][indices[i]] - activity["dist"][indices[i-1]]
        delta_t = (activity.index[indices[i]] - activity.index[indices[i-1]]).total_seconds()
        grade = 100 * (activity["alt"][indices[i]] - activity["alt"][indices[i-1]]) / (dist + 1)
        acceleration_power = 0.5 * rider_setup.total_mass * (activity["speed"][indices[i]]**2 -
            activity["speed"][indices[i-1]]**2) / delta_t
        if "temp" in activity.keys():
            temp = activity["temp"][indices[i-1]]
        if "headwind" in activity.keys():
            headwind = activity["headwind"][indices[i-1]]
        else:
            headwind = 0.0
        speed = activity["speed"][indices[i-1]:indices[i]].mean()
        steady_power = power_from_speed(rider_setup, grade, speed, altitude=altitude, temp=temp, headwind=headwind)
        pwr = steady_power + acceleration_power
        if (pwr < 0):
            pwr = 0
        activity.loc[activity.index[indices[i-1]:indices[i]], key_est] = pwr
