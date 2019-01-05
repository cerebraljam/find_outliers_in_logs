import pandas as pd
import random
import numpy as np
from datetime import datetime, date, time, timedelta
import os.path


"""
Real distribution can be extracted from BigQuery with a command like this:

WITH
  initial as (
    SELECT time, user, path, status FROM `access_log` WHERE user != 0
  ),
  identity AS (
    SELECT user, min(time) as mt FROM initial GROUP BY user ORDER BY user, mt
  ),
  single AS (
    SELECT initial.user, path, status FROM initial JOIN identity ON initial.user = identity.user
  ),
  lagged AS (
    SELECT user, path, status, LAG(path) OVER(ORDER BY path) AS path_prev FROM single GROUP BY user, path, status
  )
SELECT path_prev, path, status, count(status) as hit FROM lagged GROUP BY path_prev, path, status

"""
file='results-20190105-135944.csv'

"""
WITH
  period AS (
    SELECT * FROM `access_log`
  ),
  first AS (
    SELECT user, min(time) as start FROM period GROUP BY user
  )
SELECT period.path, count(period.path) as c FROM period JOIN first ON period.user = first.user AND period.time = first.start GROUP BY period.path ORDER BY c DESC
"""
startfile='start_action.csv'

"""
WITH
  period AS (
    SELECT * FROM `access_log`
  ),
  last AS (
    SELECT user, max(time) as finish FROM period GROUP BY user
  )
SELECT period.path, count(period.path) as c FROM period JOIN last ON period.user = last.user AND period.time = last.finish GROUP BY period.path ORDER BY c DESC
"""
endfile='end_action.csv'



observed_distribution = {}
start_distribution = {}
success = 200

def normalize_distribution(dist):
    profile = {}
    for pp in dist:
        if pp not in profile.keys():
            profile[pp] = {}
        total = 0
        for p in dist[pp]:
            total += dist[pp][p]['success']
            total += dist[pp][p]['fail']
        for p in dist[pp]:
            if dist[pp][p]['success'] > 0:
                s = p + ':success'
                profile[pp][s] = dist[pp][p]['success'] / total
            if dist[pp][p]['fail'] > 0:
                f = p + ':fail'
                profile[pp][f] = dist[pp][p]['fail'] / total

    return profile

def check_if_ok():
    if os.path.isfile(file) and os.path.isfile(startfile) and os.path.isfile(endfile):
        return True
    else:
        return False

user_profile = {}

if os.path.isfile(file) and os.path.isfile(startfile) and os.path.isfile(endfile):
    data = pd.read_csv(file, delimiter=',', header=0, encoding='utf-8', na_filter=False)

    for index, row in data.iterrows():
        if row['path_prev'] not in observed_distribution.keys():
            observed_distribution[row['path_prev']] = {}
        if row['path'] not in observed_distribution[row['path_prev']].keys():
            observed_distribution[row['path_prev']][row['path']] = {
                'success': 0,
                'fail': 0
            }
        if row['status'] == success:
            observed_distribution[row['path_prev']][row['path']]['success'] += row['hit']
        else:
            observed_distribution[row['path_prev']][row['path']]['fail'] += row['hit']
    user_profile = {'unknown': normalize_distribution(observed_distribution)}


    startdata = pd.read_csv(startfile, delimiter=',', header=0, encoding='utf-8', na_filter=False)
    total = startdata['c'].sum()
    startdata['percent'] = startdata['c'] / total

    enddata = pd.read_csv(endfile, delimiter=',', header=0, encoding='utf-8', na_filter=False)
    total = enddata['c'].sum()
    enddata['percent'] = enddata['c'] / total

    for role in user_profile:
        for action in user_profile[role]:
            total = 0
            for follow in user_profile[role][action]:
                total+= user_profile[role][action][follow]
            if (1 > round(total, 4) and total > 0) or round(total,4) > 1:
                print(role,action,total, 1-total)


user_distribution = {
    "unknown": 1
}

user_velocity = {
    "unknown": 40
}

user_lookup = {}

def generate_userlist(existing_users, nb_users):
    todays_users = existing_users[:]

    for i in range(nb_users):
        todays_users.append(random.choices(list(user_distribution.keys()), list(user_distribution.values()))[0])

    return todays_users

def generate_logs(todays_users, start_time):
    state = [0] * len(todays_users)
    next_actions = [random.randint(0,86400) for x in range(len(todays_users))]

    max_moves = [random.randint(10,200)] * len(todays_users)
    state_move = [0] * len(todays_users)
    logs = []

    for i in range(len(todays_users)):
        u = todays_users[i]
        state[i] = random.choices(list(startdata['path']), list(startdata['percent']))[0]
        user_lookup[todays_users[i] + str(i)] = todays_users[i]

    while min(next_actions) < 86400:
        ind = np.argmin(next_actions)

        if state[ind] not in user_profile[todays_users[ind]].keys():
            state[ind] = 'end'

        if state_move[ind] < max_moves[ind] and state[ind] in user_profile[todays_users[ind]].keys() and state[ind] != 'end':
            state_move[ind] += 1
            population = list(user_profile[todays_users[ind]][state[ind]].keys())
            weights = list(user_profile[todays_users[ind]][state[ind]].values())
            next_action = random.choices(population, weights)[0]

            spl = next_action.split(":")
            path = spl[0]
            status = 'success'
            if len(spl) > 1:
                status = spl[1]

            entry = [str(start_time + timedelta(seconds=next_actions[ind])), todays_users[ind] + str(ind), path, status, ind, todays_users[ind]]

            next_actions[ind] += random.randint(1, user_velocity[todays_users[ind]])
            state[ind] = path

            logs.append(entry)

        else:
            next_actions[ind] = 86400

    return logs

def align_profiles(profile1, profile2):
    if profile1.keys() != profile2.keys():
        for k in profile1.keys():
            if k not in profile2.keys():
                profile2[k] = 0.0
        for k in profile2.keys():
            if k not in profile1.keys():
                profile1[k] = 0.0
    return np.array([value for (key, value) in sorted(profile1.items())]), np.array([value for (key, value) in sorted(profile2.items())])



def cheat_compare_users(user1, user2, score, limit):
    u1, u2 = align_profiles(score[user1], score[user2])
    # u1 = np.array(list(score[user1].values()))
    # u2 = np.array(list(score[user2].values()))

    px = [1/np.power(2,x) for x in u1]
    qx = [1/np.power(2,x) for x in u2]

    p = np.array(qx)/np.array(px)
    q = np.array(px)/np.array(qx)

    real = user_lookup[user1]
    test_user_type = user_lookup[user2]

    kldp = (qx * np.log2(p)).sum()
    kldq = (px * np.log2(q)).sum()

    t = kldp < limit and kldp >= -limit and kldq < limit and kldq >= -limit
    result = { 'test': t, 'real': real == test_user_type, 'kldp': kldp, 'kldq': kldq }
    return result

def cheat_calculate_hit_rate(data, score, limit = 7):
    user_types = list(data['realtype'].unique())
    flat_status = {
        True: { True: 0, False: 0 },
        False: { True: 0, False: 0 }
    }
    for ut1 in user_types:
        all_ut1 = list(data.loc[(data['realtype'] == ut1)]['user'].unique())
        np.random.shuffle(all_ut1)
        for ut2 in user_types:
            all_ut2 = list(data.loc[(data['realtype'] == ut2)]['user'].unique())

            for j in all_ut1[:10]:
                np.random.shuffle(all_ut2)
                for i in all_ut2[:10]:
                    result = cheat_compare_users(j, i, score, limit)
                    flat_status[result['test']][result['real']] += 1

    tp = flat_status[True][True]
    fp = flat_status[True][False]

    tn = flat_status[False][True]
    fn = flat_status[False][False]

    pdenum = tp+fp
    ndenum = tn+fn

    if pdenum == 0:
        pdenum = 1
    if ndenum == 0:
        ndenum = 1

    flatlookup = {
        True: { True: tp/pdenum, False: tn/ndenum},
        False: { True: fp/pdenum, False: fn/ndenum}
    }
    return flat_status, flatlookup

def cheat_user_lookup(name):
    if name in user_lookup.keys():
        print(user_lookup[name])
        return user_lookup[name]
    else:
        return "unknown"

def cheat_lookup_all_users(list_to_lookup):
    stats = {}

    for p in list_to_lookup:
        ptype = user_lookup[p]
        if ptype not in stats.keys():
            stats[ptype] = 0
        stats[ptype] += 1
    return stats
