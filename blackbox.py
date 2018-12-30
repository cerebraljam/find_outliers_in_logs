import random
import numpy as np
from datetime import datetime, date, time, timedelta

user_distribution = {
    "buyer": 0.49, "merchant": 0.49, "bot": 0.003, "spammer": 0.003, "fraudster": 0.002, "attacker": 0.002 # in percentage
}

user_velocity = {
    "buyer": 40, "merchant": 50, "bot": 5, "spammer": 10, "fraudster": 15, "attacker": 5 # seconds per actions
}

user_start_action = {
    "buyer": 'home', "merchant": 'home', "bot": 'home', "spammer": 'home', "fraudster": 'home', "attacker": 'login:fail' # seconds per actions
}

user_profile = {
    "buyer": {
        "home": { "login:success": 0.964, "login:fail": 0.03, "password_reset": 0.005, "end": 0.001},
        "login:success": { "view_item:success": 0.978, "view_profile": 0.001, "buy_item:success":0.02, "buy_item:fail": 0.001},
        "login:fail": {"login:success": 0.9, "login:fail": 0.08, "password_reset": 0.02},
        "password_reset": {"login:success": 0.9, "login:fail": 0.09, "end": 0.01},
        "logout": {"end": 0.99, "home": 0.01},
        "view_item:success": {"comment:success": 0.05, "view_item:success": 0.65, "buy_item:success": 0.299, "buy_item:fail": 0.001},
        "buy_item:success": {"view_item:success": 0.409, "buy_item:success": 0.2, "buy_item:fail": 0.001, "logout": 0.29, "end": 0.1},
        "buy_item:fail": {"buy_item:fail": 0.01, "view_profile": 0.2, "payment_modify:success": 0.59, "payment_modify:fail": 0.1, "logout":0.05, "end": 0.05},
        "view_profile": { "update_email:success": 0.1, "update_email:fail": 0.05,
                         "update_address:success": 0.2, "update_address:fail": 0.05,
                         "payment_modify:success": 0.1, "payment_modify:fail": 0.05,
                         "view_profile": 0.05, "view_item:success": 0.4},
        "update_email:success": {"view_profile":1},
        "update_email:fail": {"update_email:success": 0.9, "update_email:fail": 0.01, "view_profile":0.09},
        "update_address:success": {"view_profile":1},
        "update_address:fail": {"update_address:success": 0.9, "update_address:fail": 0.01, "view_profile":0.09},
        "payment_modify:success": {"view_profile":1},
        "payment_modify:fail": {"payment_modify:success": 0.9, "payment_modify:fail": 0.01, "view_profile":0.09},
        "comment:success": {"view_item:success": 0.6, "buy_item:success": 0.399, "buy_item:fail": 0.001},
        "end": {}
    },
    "merchant": {
        "home": { "login:success": 0.97, "login:fail": 0.024, "password_reset": 0.005, "end": 0.001},
        "login:success": { "view_item:success": 0.978, "view_profile": 0.001, "sell_item:success":0.02, "sell_item:fail": 0.001},
        "login:fail": {"login:success": 0.9, "login:fail": 0.08, "password_reset": 0.02},
        "password_reset": {"login:success": 0.9, "login:fail": 0.09, "end": 0.01},
        "logout": {"end": 0.99, "home": 0.01},
        "view_item:success": {"view_item:success": 0.4, "sell_item:success": 0.599, "sell_item:fail": 0.001},
        "sell_item:success": {"view_item:success": 0.4, "sell_item:success": 0.399, "sell_item:fail": 0.001, "logout": 0.1, "end": 0.1},
        "sell_item:fail": {"sell_item:fail": 0.1, "view_profile": 0.2, "bank_modify:success": 0.5, "bank_modify:fail": 0.1, "logout":0.05, "end": 0.05},
        "view_profile": { "update_email:success": 0.1, "update_email:fail": 0.05,
                         "update_address:success": 0.2, "update_address:fail": 0.05,
                         "bank_modify:success": 0.1, "bank_modify:fail": 0.05,
                         "view_profile": 0.05, "view_item:success": 0.4},
        "update_email:success": {"view_profile":1},
        "update_email:fail": {"update_email:success": 0.9, "update_email:fail": 0.01, "view_profile":0.09},
        "update_address:success": {"view_profile":1},
        "update_address:fail": {"update_address:success": 0.9, "update_address:fail": 0.01, "view_profile":0.09},
        "bank_modify:success": {"view_profile":1},
        "bank_modify:fail": {"bank_modify:success": 0.9, "bank_modify:fail": 0.01, "view_profile":0.09},
        "end": {}
    },
    "bot": {
        "home": {"login:success": 0.95, "login:fail": 0.05},
        "login:success": {"view_item:success": 0.95, "view_item:fail": 0.05},
        "login:fail": {"login:success": 0.29, "login:fail": 0.7, "end": 0.01},
        "logout": {"end": 0.2, "home": 0.8},
        "view_item:success": {"view_item:success": 0.99, "view_item:fail": 0.01},
        "view_item:fail": {"view_item:success": 0.29, "view_item:fail": 0.7, "end": 0.01},
        "end": {}
    },
    "spammer": {
        "home": { "login:success": 0.7, "login:fail": 0.2,"password_reset": 0.099, "end": 0.001},
        "login:success": { "view_item:success": 0.9, "view_profile": 0.1},
        "login:fail": {"login:success": 0.7, "login:fail": 0.2, "password_reset": 0.1},
        "password_reset": {"login:success": 0.8, "login:fail": 0.1, "end": 0.1},
        "logout": {"end": 0.9, "home": 0.1},
        "view_item:success": {"comment:success": 0.5, "view_item:success": 0.4, "view_item:fail": 0.1},
        "view_item:fail": {"view_item:success": 0.19, "view_item:fail": 0.7, "logout": 0.1, "end": 0.01},
        "view_profile": { "update_email:success": 0.3, "update_email:fail": 0.1,
                         "view_profile": 0.2, "view_item:success": 0.4},
        "update_email:success": {"view_profile":1},
        "update_email:fail": {"update_email:success": 0.5, "update_email:fail": 0.4, "view_profile":0.1},
        "update_address:success": {"view_profile":1},
        "comment:success": {"view_item:success": 0.9, "logout": 0.05, "end": 0.05},
        "end": {}
    },
    "fraudster": {
        "home": { "login:success": 0.5, "login:fail": 0.45, "end": 0.05},
        "login:success": { "view_profile": 0.7, "logout":0.2, "end": 0.1},
        "login:fail": {"login:success": 0.7, "login:fail": 0.2, "end": 0.1},
        "logout": {"end": 0.9, "home": 0.1},
        "view_item:success": {"view_item:success": 0.6, "buy_item:success": 0.35, "buy_item:fail": 0.05},
        "buy_item:success": {"view_item:success": 0.3, "buy_item:success": 0.4, "buy_item:fail": 0.1, "logout": 0.1, "end": 0.1},
        "buy_item:fail": {"buy_item:fail": 0.1, "view_profile": 0.2, "payment_modify:success": 0.3, "payment_modify:fail": 0.3, "logout":0.05, "end": 0.05},
        "view_profile": { "update_email:success": 0.05, "update_email:fail": 0.05,
                         "update_address:success": 0.2, "update_address:fail": 0.05,
                         "payment_modify:success": 0.2, "payment_modify:fail": 0.2,
                         "view_profile": 0.05, "buy_item:success": 0.2},
        "update_email:success": {"buy_item:success":0.9, "view_profile": 0.1},
        "update_email:fail": {"update_email:success": 0.5, "update_email:fail": 0.4, "view_profile":0.1},
        "update_address:success": {"buy_item:success":0.9, "view_profile": 0.1},
        "update_address:fail": {"update_address:success": 0.6, "update_address:fail": 0.3, "view_profile":0.1},
        "payment_modify:success": {"buy_item:success":0.9, "view_profile": 0.1},
        "payment_modify:fail": {"payment_modify:success": 0.5, "payment_modify:fail": 0.4, "view_profile":0.1},
        "end": {}
    },
    "attacker": {
        "home": {"login:success": 0.05, "login:fail": 0.85, "end": 0.1},
        "login:success": { "logout": 0.95, "end": 0.05 },
        "login:fail": {"login:success": 0.05, "login:fail": 0.85, "end": 0.1},
        "logout": {"end": 0.1, "home": 0.9},
        "end": {}
    }
}

user_lookup = {}

def distribution():
    for role in user_profile:
        for action in user_profile[role]:
            total = 0
            for follow in user_profile[role][action]:
                total+= user_profile[role][action][follow]
            if (1 > round(total, 4) and total > 0) or round(total,4) > 1:
                print(role,action,total, 1-total)
    return {
        'user_distribution': user_distribution,
        'user_velocity': user_velocity,
        'user_start_action': user_start_action,
        'user_profile': user_profile,
        'user_lookup': user_lookup
    }

def cheat_lookup_all_users(list_to_lookup):
    stats = {}

    for p in list_to_lookup:
        ptype = user_lookup[p]
        if ptype not in stats.keys():
            stats[ptype] = 0
        stats[ptype] += 1
    return stats

def cheat_user_lookup(name):
    if name in user_lookup.keys():
        print(user_lookup[name])
        return user_lookup[name]
    else:
        return "unknown"

def generate_userlist(nb_users, existing_users):
    todays_users = existing_users[:]

    for i in range(nb_users):
        todays_users.append(random.choices(list(user_distribution.keys()), list(user_distribution.values()))[0])

    return todays_users

def generate_logs(todays_users, start_time):
    state = [0] * len(todays_users)
    next_actions = [random.randint(0,86400) for x in range(len(todays_users))]
    logs = []

    for i in range(len(todays_users)):
        u = todays_users[i]
        state[i] = user_start_action[u]
        user_lookup[todays_users[i] + str(i)] = todays_users[i]

    while min(next_actions) < 86400:
        ind = np.argmin(next_actions)
        if state[ind] != 'end':
            population = list(user_profile[todays_users[ind]][state[ind]].keys())
            weights = list(user_profile[todays_users[ind]][state[ind]].values())
            next_action = random.choices(population, weights)[0]

            spl = next_action.split(":")
            path = spl[0]
            status = 'success'
            if len(spl) > 1:
                status = spl[1]

            entry = [str(start_time + timedelta(seconds=next_actions[ind])), todays_users[ind] + str(ind), path, status, ind, todays_users[ind]]
            state[ind] = next_action

            next_actions[ind] += random.randint(1, user_velocity[todays_users[ind]])
            state[ind] = next_action
            logs.append(entry)

        else:
            next_actions[ind] = 86400


    return logs

np.seterr(divide='ignore', invalid='ignore', over='ignore')

def cheat_compare_users(user1, user2, score):
    u1 = np.array(list(score[user1].values()))
    u2 = np.array(list(score[user2].values()))
    limit = 7

    px = [1/np.power(2,x) for x in u1]
    qx = [1/np.power(2,x) for x in u2]

    p = np.array(qx)/np.array(px)
    q = np.array(px)/np.array(qx)

    real = user_lookup[user1]
    test_user_type = user_lookup[user2]

    dklp = (qx * np.log2(p)).sum()
    dklq = (px * np.log2(q)).sum()

    t = dklp < limit and dklp >= -limit and dklq < limit and dklq >= -limit
    result = { 'test': t, 'real': real == test_user_type, 'dklp': dklp, 'dklq': dklq }
    return result

def cheat_calculate_hit_rate(data, score):
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
                    result = cheat_compare_users(j, i, score)
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
