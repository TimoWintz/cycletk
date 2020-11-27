import pandas as pd
from cycletk.activities import Activity
from cycletk import activities, riders
from opendata import OpenData

def calendar_from_golden_cheetah(athlete_id, wp=20000, estimate_pwr=False):
    od = OpenData()
    athlete = od.get_local_athlete(athlete_id)

    gc_activities = athlete.activities()
    activity_list = []
    rider_list = []
    for activity in list(gc_activities)[:5]:
        cp = activity.metadata['METRICS']['cp_setting']
        weight = activity.metadata['METRICS']['athlete_weight']
        rider = riders.Rider(cp=float(cp), wp=float(wp), weight=float(weight))
        activity_list.append(activities.read_golden_cheetah(athlete_id, activity.id))
        rider_list.append(rider)
    return load_activities(rider_list, activity_list, estimate_pwr=estimate_pwr)

def load_activities(riders, activities, estimate_pwr=False):
    if type(riders) == list:
        lines = [Calendar.line(riders[i], activities[i], estimate_pwr) for i in range(len(activities))]
    else:
        lines = [Calendar.line(riders, activities[i], estimate_pwr) for i in range(len(activities))]
    index, lines = zip(*lines)
    x = Calendar(lines, index = index)
    x.sort_index(inplace=True)
    return x
        
class Calendar(pd.DataFrame):

    @staticmethod
    def line(rider, activity, estimate_pwr=False):
        idx = activity.start
        pz = activity.power_zones(rider)
        res = {}
        if "dist" in activity.keys():
            res
        res = {
            "rider_cp" : rider.cp,
            "rider_wp" : rider.wp,
            'rider_weight' : rider.weight,
            "duration" : activity.duration(),
            "distance" : activity.distance(),
            "average_pwr" : activity.average_power(),
            "normalized_pwr" : activity.normalized_power(),
            "stress" : activity.stress(rider),
            "activity_source_type" : activity.source_type,
            "activity_source" : activity.source
        }
        for i in rider.PZ_NAMES:
            res[i] = pz["time"][i]
        return idx, res

    def get_activity(self, index):
        if self["activity_source_type"][index] == "file":
            return activities.read(self["activity_source"][index])
        elif self["activity_source_type"][index] == "golden_cheetah":
            return activities.read_golden_cheetah(*self["activity_source"][index].split(","))