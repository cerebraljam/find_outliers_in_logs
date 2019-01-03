
# How to group users and identify outliers in a log stream, based on behaviour

## What is demonstrated in this notebook

In this notebook I demonstrate, from a (generated) stream of logs, how to group users by behaviour, and how to highlight which behaviour group an analyst should focus his attention on. 

This method is good when monitoring a stable and closed system, with a finite number of possible actions and known definition of success and failure.

Probabilities are based on a "Subject (unique key), Action (ex.: login), Outcome (success/fail)" discrete format. Continuous Action variables would need to be reduced to discrete output to fit this model.

## Why is this a hard problem?

What is a normal action? Which one is suspect? The answer is "it depends". 

To group individual together, we need a way to classify each action taken over time, and somehow group them together. A clustering function like KMean requires us to know how many cluster we have, and we need mesurable features to group them together. Normal clustering functions lose in efficiency the more we add features.

Machine learning techniques are also great to identify patterns over a large number of repeating features, but struggle to handle unusual events especially when data is skewed by a high volume of "normal" actions. This is why in text analysis we normally remove common [English stop words](https://www.ranks.nl/stopwords). But what if these stop words were necessary to identfy an outlier? How can we keep them without debalancing everything?

One other hypothesis would be to use Hidden Markov Chains, but for HNN, we need classified data. Sequences also need to be somehow of a consistent length, but users are most likely not to follow a straight pattern from A to B, which complicates the creation of HMM.

Another approach would be to use rules. Classically, Intrusion detection systems recognize a set of patterns, and uses thresholds to know when to raise a flag, generating a large number of false positive. A rule based system is also unlikely to be able to detect the usage of normal functions in an abusive way.

In the end, we need each features from all possible events to be classified and scored somehow, and we need a way to group these scores together.

## Approach taken

### What this notebook is the following

* Step 1: We calculate the probability of each sequences of actions being taken by all users over a defined period of time (ex.: 1 day). From these probabilities, we create a lookup table for each possible sequence.
* Step 2: For each users, we sum the probability (using logarithm base 2) for each action taken. Depending on the likelyhood of each action, this will create different profiles per user.
* Step 3: Using the profile created on Step 2, we group them together by calculating the "distance" between all actions. If the distance is small, then the users have similar behaviour profile. If the distance is far, then they have different behaviour.
* Step 4: With all users grouped together, we can quickly analyse a group of user and label that group with a usage profile. This profile shouldn't change over time. If new behaviours are observed, new groups will be created. An analyst then only have to pay attention to the new group and groups identified as malicious when monitoring a system.

### Caviats

* For Step 1, this can be done on a live stream of logs, but for systems generating a large quantify of logs, this could result in integer overflow. Using rolling windows (ex.: calculate for a period, then use these probabilities distribution to analyse the next period, while creating a new distribution for the next period) might be a better solution.
* For Step 2, this works better when there is a clear start and end of sequence for a user. As it is coded right now, scoring a partial session would end up creating a large quantity of behaviour group, which could be noisy. However, this could be interesting to identify a list of "opening moves".
* For Step 3, the "distance" between each users is somewhat arbitrary selected. If we impose a short maximum distance (ex.: 5, 0 being perfectly identical), then this method will generate a large number of groups to represent all possible behaviours, but if we use a larger maximum distance (ex.: 10), then we increase the likelihood of having misclassifications and group outlier with normal users.
* For Step 4: None here :-D This is where I would start configuring rules or send data to a machine learning model, because users are now classified in limited groups.

## The theory behind the approach

Information Theory is normally used for signal analysis and compression. By using Surprisal Analysis and Kullback–Leibler divergence, we can group similar behaviour togethers.

* [Information Content](https://en.wikipedia.org/wiki/Information_content)
* [Surprisal Analysis](https://en.wikipedia.org/wiki/Surprisal_analysis)
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
* [Example of Surprisal](http://www.umsl.edu/~fraundorfp/egsurpri.html)

Basically, Surprisal Analysis is a way to measure (in bits) how surprised we should be when looking at a specific behaviour. 

For example:
* There is 1/6 chance to roll a 6 on a d6 die, gives us -log2(1/6) ~= 2.4849 bits of information. If you are playing D&D, you will be happily surprised.
* Not rolling a 6 on a die represents -log2(5/6) ~= 0.263 bits. So rolling anything but a 6 in that same D&D game will most likely leave you dissapointed. 
* The chance of rolling 10 x 6 in a row on a die is 1/60466176 (10 * -log2(1/6) ~= 25.85 bits). This is highly unlikely.

So later when we look at the surprisal score of a user and we see 51550 (bits), this tells us that what this user did is really, really unlikely.

This also tells us that if the total surprisal score of the actions of a user is low, then we can safely assume that the actions taken by that user are normal.

Surprisal are calculated using Log base 2, which has the nice particularity to be additive instead of being multiplied like probabilities, keeping numbers relatively small and easy to manipulate.

This is what this notebook intend to demonstrate: Using Surprisal analysis, we will assign a score to actions, and by adding up the score of each action, identify series of actions that are unlikely to occur.


# Step 0: We need logs

What is explained in this notebook can be applied to real logs, but for the experimentation, I generate logs for simulated users based on a determined probability distribution.

## Caviat: I use some cheats to demonstrate how the notebook works...

But these cheats are not necessary to make this technique work. However, I did use that cheat at first to identify the most efficient "distance" to use in Step 3. 


## Different User Profiles Generated By The Library

The following profiles are being generated.

Normal users
* Buyer
* Merchants

Abnormal users:
* Scraper bots
* Spammers
* fraudster
* Account Attackers

Buyers and merchants represent 98% of our logs. Leaving 2% to the abnormal users. However, the actions taken by each users being on a probability distribution. 

It is possible to see an "attakcer" user being classified as a user, because that attacked might have had a change of heart and didn't attack after all. This allows us to demonstrate that this approach is not perfect: a small quantity of unlikely actions will most probably pass under the radar if we just look at the score of each user. That attacker might however be identified if the actions doesn't match the ones of normal users, which would generate a new group and single him/her out.



```python
## This cell initialize the libraries needed for this notebook, 
## along with the blackbox functions used to generate the users and logs

import random
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from math import log, pow

%load_ext autoreload
%autoreload 2

from blackbox import distribution as user_profile_distribution
from blackbox import generate_userlist, generate_logs, cheat_calculate_hit_rate, cheat_lookup_all_users

magic = user_profile_distribution()
```

# Step 0.1: Generating The User Database

The following cells generates our users.

Note: I often reinitializing the random seed, just to keep the testing consistence. Setting *random_seed* to False will generate new data every time.


```python
## Initial number of users in our system
number_of_daily_users = 3000 # The more we add users, the longer the notebook takes to run.
```


```python
random_seed = 42

if random_seed:
    random.seed(random_seed)

all_user_lists = [] # Later on, we can add users to our list by supplying it to the generate_userlist function

## We define how many new users to add
number_of_new_users = number_of_daily_users - len(all_user_lists)

all_user_lists = generate_userlist(all_user_lists, number_of_new_users)
todays_user_lists = random.sample(all_user_lists, number_of_daily_users)

print(len(todays_user_lists), 'users in the database. Type of the 15 firsts:', todays_user_lists[:15])
```

    3000 users in the database. Type of the 15 firsts: ['buyer', 'buyer', 'merchant', 'merchant', 'merchant', 'buyer', 'merchant', 'buyer', 'buyer', 'merchant', 'merchant', 'spammer', 'merchant', 'merchant', 'buyer']


# Step 0.2: Generating Logs For Day 1

Note: The more users we have, the more log events will be generated. The probability distribution of each user ensures that they will start with a defined action, crawl the site following a defined pattern, and logout eventually, until the end of the day.


```python
%%time
if random_seed:
    random.seed(random_seed)

start_time = datetime(2019,1,1,0,0)
day1_logs = generate_logs(todays_user_lists, start_time)

print(len(day1_logs), 'log events generated for', len(todays_user_lists), 'users')
```

    69095 log events generated for 3000 users
    CPU times: user 17.8 s, sys: 87 ms, total: 17.9 s
    Wall time: 18 s


## Transforming the logs in a pandas dataframe (for this notebook...)

The transition surprisal lookup table used in this notebook calculates scores based on the movements of the users between each actions. For example:

* login -> view_items (success) will result in a low surpisal value
* login -> buy_item (success) never happened. If this sequence happen, this should be a huge red flag.

Notice that there is one level skipped here: the status of the previous path isn't taken into consideration. Adding it would definitely make the calculations more sensitive to anomalies, but with the cost of an increase on complexity.


```python
def transform_logs_to_pandas(logs):
    data = pd.DataFrame(np.array(logs), columns=['time', 'user', 'path', 'status', 'uidx', 'realtype'])
    
    data['prev_path'] = data.groupby(['user'])['path'].shift(1)
    data['prev_path'] = data['prev_path'].fillna("")

    data['prev_status'] = data.groupby(['user'])['status'].shift(1)
    data['prev_status'] = data['prev_status'].fillna("")
    return data
    
day1_data = transform_logs_to_pandas(day1_logs)


# Example of failed actions in the logs. uidx and realtype are "cheat" columns, and not necessary in a real case usage.

print(day1_data.loc[(day1_data['path'] == 'login') & (day1_data['status'] == 'fail')].head())
```

                        time          user   path status  uidx  realtype  \
    2    2019-01-01 00:01:14      buyer166  login   fail   166     buyer   
    4    2019-01-01 00:01:35      buyer166  login   fail   166     buyer   
    200  2019-01-01 00:14:55  attacker2467  login   fail  2467  attacker   
    202  2019-01-01 00:15:00  attacker2467  login   fail  2467  attacker   
    204  2019-01-01 00:15:01  attacker2467  login   fail  2467  attacker   
    
        prev_path prev_status  
    2                          
    4       login        fail  
    200                        
    202     login        fail  
    204     login        fail  


## Step 1 : Generate the transition lookup table

**This is where the magic happens.**

The following cell generates the transition lookup table used to score each actions taken by the users.

The format is as follow:

```
['current path'],['previous path']: {
    'fail': 0, # How many time this action transition failed. (ex. View Items: Success, from: Login: Success)
    'success': 13, # How many time this action transition succeded (ex. View Items: Fail, from: Login: Success)
    'fsurprisal': 11.266786540694902, # Surprisal value if there is a failure happens
    'ssurprisal': 7.56634682255381 # Surprisal value if that action is successful.
    }
```

The surprisal value is directly related to the likelihood of an actions happening. If an actions is observed successfully few million times, then the successful surprisal value will be really low. However, the failure surprisal will be much higher if it never happens.


```python
def init_transition_surprisal_lookup(data, key, prev_key, feature, success):
    surprisal = {}

    for pkey in data[key].unique():
        data_for_pkey = data.loc[(data[key] == pkey)]
        denum = len(data.loc[(data[key] == pkey)])

        for ppkey in data_for_pkey[prev_key].unique():
            ds = data_for_pkey.loc[(data_for_pkey[prev_key] == ppkey) & (data_for_pkey[feature] == success)]
            df = data_for_pkey.loc[(data_for_pkey[prev_key] == ppkey) & (data_for_pkey[feature] != success)]

            dsuccess = len(ds) * 1.0
            dfail = len(df) * 1.0

            if dsuccess == 0:
                dsuccess = 1.0 

            if dfail == 0:
                dfail = 1.0

            if (pkey not in surprisal.keys()):
                surprisal[pkey] = {}

            surprisal[pkey][ppkey] = {
                'success': len(ds), 
                'fail': len(df), 
                'ssurprisal': log(1/(dsuccess / denum),2), # Magic!!
                'fsurprisal': log(1/(dfail / denum),2), # Magic!!
            }
    return surprisal

transition_surprisal = init_transition_surprisal_lookup(day1_data, 'path', 'prev_path', 'status', 'success')
```

The next cell creates a wrapper function for the transition surprisal lookup table. If the sequence was previously observed, it will return it, and will return the probability of 1 over the total number of actions taken (unlikely) for unobserved events, making them stand out.


```python
def get_transition_surprisal(path, prev_path, surprisal, data):
    if path not in list(surprisal.keys()):
        denum = len(data)
        return {
            'fail': 0,
            'success': 0,
            'ssurprisal': log(1/(1/denum),2),
            'fsurprisal': log(1/(1/denum),2),
        }
    else:
        if prev_path not in list(surprisal[path].keys()):
            denum = len(data.loc[(data['path'] == path)])
            return {
                'fail': 0,
                'success': 0,
                'ssurprisal': log(1/(1/denum),2),
                'fsurprisal': log(1/(1/denum),2),
            }
        else:
            return surprisal[path][prev_path]

# Usage example: what is the surprisal values for a "buy_item" action 
# if the user just logged in on the previous action?
get_transition_surprisal('buy_item', 'login', transition_surprisal, day1_data)
```




    {'fail': 1,
     'fsurprisal': 11.903128676812319,
     'ssurprisal': 7.095773754754716,
     'success': 28}



# Step 2: For each users, sum the probability for each action taken.

The following function takes the dataframe with all the logs for a day, and returns the sum of the surprisal values, conditional to a success or a failure, for each actions taken by a user.


```python
def get_user_transition_score(data, surprisal, key, feature, success_val):
    accumulator = {}
    key_last_path = {}
    
    for index,row in data.iterrows():
        if row[key] not in key_last_path.keys():
            key_last_path[row[key]] = ""
            
        if row[key] not in accumulator.keys():
            accumulator[row[key]] = {k:0 for k in data[feature].unique()}
            
        if row[feature] is success_val:
            accumulator[row[key]][row[feature]] += get_transition_surprisal(row[feature],key_last_path[row[key]], surprisal, data)['ssurprisal']
        else:
            accumulator[row[key]][row[feature]] += get_transition_surprisal(row[feature],key_last_path[row[key]], surprisal, data)['fsurprisal']

        key_last_path[row[key]] = row[feature]
                                    
    return accumulator


user_transition_score = get_user_transition_score(day1_data, transition_surprisal, 'user', 'path', 'success')
```

## Optional (but useful): Identify how shady are each outliers

The following three cells calculate the mean and standard deviation of each users, and display the users with a score of 2 or more standard deviations.


```python
cumulative_score = [[v,sum(user_transition_score[v].values())] for v in [k for k in list(user_transition_score.keys())]]

df_cumulative_score = pd.DataFrame(cumulative_score, columns=['user', 'surprisal'])

avg = df_cumulative_score['surprisal'].mean()
std = df_cumulative_score['surprisal'].std()
df_cumulative_score['z'] = (df_cumulative_score['surprisal'] - avg) / std

```


```python
# List all users with a zscore over 2.
df_cumulative_score.loc[df_cumulative_score['z'] >= 2].sort_values(by=['surprisal'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>surprisal</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1451</th>
      <td>bot1392</td>
      <td>34192.771972</td>
      <td>30.525805</td>
    </tr>
    <tr>
      <th>2407</th>
      <td>bot831</td>
      <td>29638.440077</td>
      <td>26.438511</td>
    </tr>
    <tr>
      <th>910</th>
      <td>bot524</td>
      <td>22653.550349</td>
      <td>20.169908</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>bot2148</td>
      <td>18399.108431</td>
      <td>16.351751</td>
    </tr>
    <tr>
      <th>529</th>
      <td>bot202</td>
      <td>17577.598547</td>
      <td>15.614486</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>bot1387</td>
      <td>17495.991605</td>
      <td>15.541248</td>
    </tr>
    <tr>
      <th>870</th>
      <td>bot194</td>
      <td>8600.834908</td>
      <td>7.558272</td>
    </tr>
    <tr>
      <th>315</th>
      <td>bot1972</td>
      <td>8094.871867</td>
      <td>7.104195</td>
    </tr>
    <tr>
      <th>502</th>
      <td>bot691</td>
      <td>7153.671801</td>
      <td>6.259513</td>
    </tr>
    <tr>
      <th>76</th>
      <td>bot1183</td>
      <td>6756.518015</td>
      <td>5.903087</td>
    </tr>
  </tbody>
</table>
</div>




```python
# List the 10 most boring users.
df_cumulative_score.sort_values(by=['surprisal'], ascending=False).tail(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>surprisal</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2226</th>
      <td>buyer966</td>
      <td>20.231532</td>
      <td>-0.142406</td>
    </tr>
    <tr>
      <th>222</th>
      <td>attacker1130</td>
      <td>16.531092</td>
      <td>-0.145727</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>attacker721</td>
      <td>16.531092</td>
      <td>-0.145727</td>
    </tr>
    <tr>
      <th>1834</th>
      <td>attacker1073</td>
      <td>16.531092</td>
      <td>-0.145727</td>
    </tr>
    <tr>
      <th>120</th>
      <td>fraudster1293</td>
      <td>11.545447</td>
      <td>-0.150202</td>
    </tr>
    <tr>
      <th>1546</th>
      <td>buyer2158</td>
      <td>11.545447</td>
      <td>-0.150202</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>buyer1840</td>
      <td>11.545447</td>
      <td>-0.150202</td>
    </tr>
    <tr>
      <th>2904</th>
      <td>merchant142</td>
      <td>11.545447</td>
      <td>-0.150202</td>
    </tr>
    <tr>
      <th>649</th>
      <td>buyer540</td>
      <td>11.545447</td>
      <td>-0.150202</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>buyer238</td>
      <td>4.985645</td>
      <td>-0.156089</td>
    </tr>
  </tbody>
</table>
</div>



## Cheat cell: Identify the ideal "distance" to increase the True Positive rate, and reduce the False Positive Rate

Without using this cheat, we need to guess the ideal "distance". From experimentation, this should be between 4 and 8. 9 and more will cause a higher rate of False Positive.


```python
np.seterr(divide='ignore', invalid='ignore', over='ignore')

if random_seed:
    np.random.seed(random_seed)

maxlimit = 1
maxtp = 0
mintn = 1
best_flat_lookup = {}

for l in range(2, 10):
    flat_status, flat_lookup = cheat_calculate_hit_rate(day1_data, user_transition_score, l)
    if maxtp <= flat_lookup[True][True] and mintn >= flat_lookup[True][False]:
        maxtp = flat_lookup[True][True]
        mintn = flat_lookup[True][False]
        maxlimit = l
        print('best', l, flat_lookup, maxtp, mintn, (maxtp+mintn)/2,maxlimit)
    

flat_status, flat_lookup = cheat_calculate_hit_rate(day1_data, user_transition_score, maxlimit)

print('limit', maxlimit)
print('count', True, flat_status[True])
print('count', False, flat_status[False])

print('percent', True, flat_lookup[True])
print('percent', False, flat_lookup[False])

```

    best 2 {True: {True: 0.9354838709677419, False: 0.12112308194580476}, False: {True: 0.06451612903225806, False: 0.8788769180541952}} 0.9354838709677419 0.12112308194580476 0.5283034764567733 2
    limit 2
    count True {True: 165, False: 13}
    count False {True: 380, False: 2691}
    percent True {True: 0.9269662921348315, False: 0.12373819602735266}
    percent False {True: 0.07303370786516854, False: 0.8762618039726473}



```python
# Bypassing the cheat function and forcing the maximum distance.
maxlimit = 5
```

# Step 3: Grouping users by their behaviour


```python
# Calculating the distance between profiles requires to convert log base 2 back to probabilities, 
# which sometimes causes overflow when profiles are too far apart. We can safely ignore these errors
np.seterr(divide='ignore', invalid='ignore', over='ignore') 

# If new actions are observed and are missing, we need to 1) ignore them, or 2) add them for the calculation
# This function ensure that both profiles has the same keys, and asign a value of 0 if it was missing
# Profiles and keys are returned in order of key names
def align_profiles(profile1, profile2):
    if profile1.keys() != profile2.keys():
        for k in profile1.keys():
            if k not in profile2.keys():
                profile2[k] = profile1[k]
        for k in profile2.keys():
            if k not in profile1.keys():
                profile1[k] = profile2[k]
    p1 = [value for (key, value) in sorted(profile1.items())]
    p2 = [value for (key, value) in sorted(profile2.items())]
    allkeys = [key for (key, value) in sorted(profile2.items())]
    return p1, p2, allkeys

# Magic here too!
# this function calculate the proximity between two profiles
# The Kullback-Leibler function isn't symetric, so A could be close to B, but B be absolutely different to A
# This is why we calculate the two possible distance (a/b and b/a)
# The return value is True if the two profiles are within the limit distance, else it returns False
def compare_profiles(profile1, profile2, limit = 7):
    u1, u2, trash = align_profiles(profile1, profile2)
    
    # Ref: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition
    px = [1/np.power(2,x) for x in np.array(u1)]    
    qx = [1/np.power(2,x) for x in np.array(u2)]
    
    p = np.array(qx)/np.array(px)
    q = np.array(px)/np.array(qx)
    dklp = (qx * np.log2(p)).sum()
    dklq = (px * np.log2(q)).sum()
    
    # dklp/dlklq == Divergence Kullback-Leibler for p and q
    t = (dklp < limit and dklp >= -limit) and (dklq < limit and dklq >= -limit)
    
    return {'test': t, 'dklp': dklp, 'dklq': dklq}

# Example of usage
test_users = random.sample(user_transition_score.keys(),2)
print(test_users[0], user_transition_score[test_users[0]])
print(test_users[1], user_transition_score[test_users[1]])
compare_profiles(user_transition_score[test_users[0]], user_transition_score[test_users[1]], maxlimit)
```

    merchant1159 {'login': 4.985644707022931, 'view_item': 94.19583408958562, 'comment': 0, 'buy_item': 0, 'logout': 0, 'end': 11.545447181678094, 'sell_item': 83.69866845609923, 'home': 0, 'bank_modify': 0, 'view_profile': 0, 'update_address': 0, 'update_email': 0, 'password_reset': 0, 'payment_modify': 0}
    buyer1515 {'login': 4.985644707022931, 'view_item': 67.7400193361843, 'comment': 0, 'buy_item': 53.12436221098946, 'logout': 10.867278739709663, 'end': 11.545447181678094, 'sell_item': 0, 'home': 0, 'bank_modify': 0, 'view_profile': 0, 'update_address': 0, 'update_email': 0, 'password_reset': 0, 'payment_modify': 0}





    {'dklp': 83.69285085506084, 'dklq': 63.991640950699122, 'test': False}




```python
# This function is used to calculate the average profile for a behaviour group.
# The idea is that it will align the profiles first, then calculate the average of the values for each keys
def compile_average_for_type(array):
    garbage = {}
    for i in range(len(array)):
        if len(garbage.keys()) == 0:
            garbage = array[i]
        else:
            g, a, allkeys = align_profiles(garbage, array[i])
            c = np.array(g) + np.array(a)
            for idx in range(len(allkeys)):
                garbage[allkeys[idx]] = c[idx] 
    for k in garbage.keys():
        garbage[k] = garbage[k] / len(array)
    return garbage

# Example of usage
array = [{'a':1,'b':2,'c':3}, {'a':1,'b':2,'c':3}, {'a':1,'b':2,'c':3, 'd':4}, {'a':3,'b':1,'c':4}]
compile_average_for_type(array)
```




    {'a': 1.5, 'b': 1.75, 'c': 3.25, 'd': 4.0}




```python
# Once classified, this function will add to the identified class
# If the target class does not exist yet (the classification function returns an 
# inexistant class the candidate didn't match any existing profile)
# Then the function will add the candidate to that class.
def add_candidate_to_behaviour_group(candidate_name, matching_class, behaviour_group_table):  
    if matching_class not in behaviour_group_table.keys():
        behaviour_group_table[matching_class] = []

    if candidate_name not in behaviour_group_table[matching_class]:
        behaviour_group_table[matching_class].append(candidate_name)
        
    return candidate_name
```


```python
# This is a cleanup function. If a user is already a member of a class, it will be removed 
# to be reclassified again
# If the class is empty once cleaned, it will stay emptied since another candidate 
# might have that profile later one.
def remove_from_classification(candidate_name, behaviour_group_table):
    cleaneds = [] # convention, I put s to array variables
    empties = []
    for be, be_list in behaviour_group_table.items():
        if candidate_name in behaviour_group_table[be]:
            behaviour_group_table[be].remove(candidate_name)
            cleaneds.append(be)
        if len(behaviour_group_table[be]) == 0:
            empties.append(be)
    for e in empties:
        del behaviour_group_table[e]
            
    return cleaneds       

# This function compares the candidates to the saved average calculated for each behaviour group
def classify_candidates_average(candidate_name, behaviour_type_average, score, limit = 7):
    potential_matching_type = {}
    passing_score = 0.9
    sample_size = 10
    small_size_adjustment = 2
    
    for be, be_average in behaviour_type_average.items():
        post = 0.1 # this is the prior

        result = compare_profiles(score[candidate_name], be_average, limit)
            
        if result['test'] == True:
            potential_matching_type[be] = post

    if len(potential_matching_type.keys()) == 0:
        new_class_name = max(0,len(list(behaviour_type_average.values())))
        return new_class_name
    else:
        return max(potential_matching_type, key=potential_matching_type.get)

# This is a helper function that calls all the actions in order for all unclassified users
# The candidate is selected at random from the list to prevent behaviour_class skewing if a batch of close users has
# slightly similar behaviours.
# The recalculation of the average only happens on the first value, and later one at each 4 profiles.
# Profiles selected to calculate the average are also selected at random to try to keep the average balanced over time
# This is also to prevent behaviour group polution. An earlier version was using Bayes probability update to test 
# a candidate against multiple members of a class, which had a bad side effect: If the maxmimum allowed distance is 5, 
# the first value might be 5, the second, could be between 0 and 10 and would be found as a match, then the following
# could stretch until 15, and so on. If enough tests were made, because we know that the classification is normally
# quite accurate, this would allow group to degenerate.
def classify_users_in_list(unclassified_user_lists, behaviour_group_table, behaviour_average_table, score, limit = 7):
    # select one user
    candidate_name = random.choice(unclassified_user_lists)
    if candidate_name:
        # classify user
        cleanup = remove_from_classification(candidate_name, behaviour_group_table)
        
        matching_class = classify_candidates_average(candidate_name, behaviour_average_table, score, limit)

        # add the user to the proper type
        add_candidate_to_behaviour_group(candidate_name, matching_class, behaviour_group_table)
        if len(behaviour_group_table[matching_class]) % 4 == 0 or len(behaviour_group_table[matching_class]) == 1:
            scores_for_users = random.sample(
                [score[x] for x in behaviour_group_table[matching_class]], 
                min(len(behaviour_group_table[matching_class]),10)
            )
            behaviour_average_table[matching_class] = compile_average_for_type(scores_for_users)
        unclassified_user_lists.remove(candidate_name)
```

## Extract the users to classify from the observed active users

The following cell initialize the behaviour_group_table and behaviour_average_table variables.
Then we get the list of observed users, and we create a list of unclassified users to classify


```python
if random_seed:
    random.seed(random_seed)

behaviour_group_table = {}
behaviour_average_table = {}
print(len(list(user_transition_score.keys())))
unclassified_user_lists = random.sample(list(user_transition_score.keys()), min(len(todays_user_lists), len(list(user_transition_score.keys()))))
```

    2999


The next cell is heavy to run. This is where we go through the list of unclassified users and we add them in the proper group.

Note: *behaviour_group_table* and *behaviour_average_table* are modified by the subfunctions


```python
%%time
while len(unclassified_user_lists):
    classify_users_in_list(unclassified_user_lists, behaviour_group_table, behaviour_average_table, user_transition_score, maxlimit)
```

    CPU times: user 12 s, sys: 38.9 ms, total: 12.1 s
    Wall time: 12.1 s


## Cheat cell: List all behaviour groups and the number of user by their real type

This cell is only to demonstrate that the previous functions mostly group users by their good real time.
There are some misclassifications, this could be caused by the randomness of the actions taken by the generated users. 

Also, an attacker's action could look exactly like a real user. In these cases, we will most probably won't be able to identify that that user was an attacker, unless other actions were taken that would separate the behaviour profile from a normal user.


```python
for k in behaviour_group_table.keys():
        type_average = np.mean([sum(user_transition_score[x].values()) for x in behaviour_group_table[k]])
        print(k, type_average, len(behaviour_group_table[k]), cheat_lookup_all_users(behaviour_group_table[k]))
```

    0 126.597730317 746 {'merchant': 746}
    1 93.7006802612 798 {'buyer': 798}
    2 165.106230106 274 {'buyer': 274}
    3 82.1226685229 251 {'buyer': 251}
    4 159.379639992 96 {'buyer': 96}
    5 143.359317165 731 {'merchant': 731}
    6 249.880103456 1 {'buyer': 1}
    7 191.368894111 6 {'buyer': 5, 'fraudster': 1}
    8 244.894155926 2 {'buyer': 2}
    9 249.65153609 1 {'buyer': 1}
    10 29.4321473213 2 {'merchant': 2}
    11 34192.7719717 1 {'bot': 1}
    12 365.215886043 3 {'merchant': 3}
    13 40.7688380859 11 {'buyer': 11}
    14 1092.99622918 1 {'bot': 1}
    15 184.666656114 3 {'merchant': 3}
    16 19.556956114 10 {'buyer': 2, 'attacker': 5, 'fraudster': 2, 'merchant': 1}
    17 169.155946646 5 {'spammer': 5}
    18 122.43986214 1 {'buyer': 1}
    19 209.73582516 4 {'merchant': 4}
    20 37.0202139404 3 {'buyer': 1, 'attacker': 2}
    21 192.750208925 3 {'spammer': 3}
    22 343.179525931 2 {'merchant': 2}
    23 175.363943864 3 {'merchant': 3}
    24 71.0088488556 2 {'fraudster': 2}
    25 4.98564470702 1 {'buyer': 1}
    26 6756.51801549 1 {'bot': 1}
    27 44.817450904 7 {'merchant': 7}
    28 435.842507689 2 {'buyer': 2}
    29 18399.1084312 1 {'bot': 1}
    30 29638.4400773 1 {'bot': 1}
    31 29.0612569806 4 {'merchant': 3, 'buyer': 1}
    32 169.544596314 2 {'buyer': 2}
    33 299.439673576 1 {'merchant': 1}
    34 96.9310103338 4 {'fraudster': 1, 'attacker': 3}
    35 8094.87186701 1 {'bot': 1}
    36 7153.6718007 1 {'bot': 1}
    37 97.4881028882 1 {'buyer': 1}
    38 8600.83490844 1 {'bot': 1}
    39 199.200463842 1 {'merchant': 1}
    40 118.705632495 2 {'fraudster': 1, 'buyer': 1}
    41 113.699276303 1 {'spammer': 1}
    42 219.052547518 1 {'buyer': 1}
    43 22653.5503494 1 {'bot': 1}
    44 174.878220167 1 {'attacker': 1}
    45 17495.9916045 1 {'bot': 1}
    46 11.5454471817 1 {'buyer': 1}
    47 42.4732538322 1 {'fraudster': 1}
    48 17577.5985467 1 {'bot': 1}


# Step 4: Analyse the behaviour groups, and lighlight which group contains outliers

This sections demonstrates how we can visualise and classify each group of behaviours. 


```python
# This cell creates an helper function to generate a graph of the number of users in each behaviour group

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib

def graph_user_distribution_by_behaviour_id(type_table, average_table, transition_score):  
    current_behaviour_state_table = {}
    for k in type_table.keys():
        type_average = np.mean([sum(transition_score[x].values()) for x in type_table[k]])
        current_behaviour_state_table[k] = {'type': k, 'score':type_average, 'nbmembers':len(type_table[k]), 'behaviour': average_table[k]}
    
    plt.figure(figsize=(20,5))

    index = np.array(list(current_behaviour_state_table.keys()))
    names = list(type_table.keys())
    v1 = [current_behaviour_state_table[x]['nbmembers'] for x in names]
    plt.figure(1, figsize=(9, 5))

    bar_width = 0.4
    spare_width = (1 - bar_width*2)/2

    rects1 = plt.bar(index, v1, bar_width,
                    color='b',
                    label='Nb Members')

    plt.xlabel('Behaviour Group ID')
    plt.ylabel('Number of Users')
    plt.title('Number of User Per Behaviour Group')
    plt.xticks(index)
    plt.grid()
    plt.legend()
    return plt
```


```python
if 0: #graph debugging... ignore
    current_behaviour_state_table = {}

    for k in behaviour_group_table.keys():
        type_average = np.mean([sum(user_transition_score[x].values()) for x in behaviour_group_table[k]])
        current_behaviour_state_table[k] = {'type': k, 'score':type_average, 'nbmembers':len(behaviour_group_table[k]), 'behaviour': behaviour_average_table[k]}

    surprisal_stats = np.array([
        [current_behaviour_state_table[y]['behaviour'][x]/sum(current_behaviour_state_table[y]['behaviour'].values()) for x in sorted(current_behaviour_state_table[1]['behaviour'].keys())]
        for y in list(current_behaviour_state_table.keys())# [key for (key, value) in sorted(current_behaviour_state_table.items())]
    ])

    keylists_columns = [x for x in list(current_behaviour_state_table.keys()) if current_behaviour_state_table[x]['nbmembers'] > 0]
    path_rows = sorted(current_behaviour_state_table[keylists_columns[0]]['behaviour'].keys())
    surprisal_values = list(current_behaviour_state_table[keylists_columns[0]]['behaviour'].values())
    # np.array(surprisal_values) * current_behaviour_state_table[keylists_columns[0]]['nbmembers']

    x = [[1,2,3,4],
        ['a','b','c','d'],
        [4,5,6,7]
        ]
    # print(np.array(surprisal_values))

    #     p2 = [value for (key, value) in sorted(profile2.items())]
    #     allkeys = [key for (key, value) in sorted(profile2.items())]

    print(keylists_columns)
#     print(path_rows)
#     print(surprisal_stats[0])
    # print(sorted(current_behaviour_state_table[1]['behaviour'].keys()))

```


```python
# This cell creates an helper function that displays the distribution of actions for each group.
# Note that the surprisal value for a single action is maxed out at 500. More than this, the graph
# becomes useless since all normal actions squashed by the outliers

def graph_surprisal_distribution_by_action(type_table, average_table, transition_score):
    current_behaviour_state_table = {}
    for k in type_table.keys():
        type_average = np.mean([sum(transition_score[x].values()) for x in type_table[k]])
        current_behaviour_state_table[k] = {'type': k, 'score':type_average, 'nbmembers':len(type_table[k]), 'behaviour': average_table[k]}
    
    surprisal_stats = np.array([
        [current_behaviour_state_table[y]['behaviour'][x] for x in sorted(current_behaviour_state_table[1]['behaviour'].keys())]
#         [current_behaviour_state_table[y]['behaviour'][x]/sum(current_behaviour_state_table[y]['behaviour'].values()) for x in sorted(current_behaviour_state_table[1]['behaviour'].keys())]
        for y in [x for x in list(current_behaviour_state_table.keys()) if current_behaviour_state_table[x]['nbmembers'] > 0]
    ]).T
    
    keylists_columns = [x for x in sorted(list(current_behaviour_state_table.keys())) if current_behaviour_state_table[x]['nbmembers'] > 0]
    if len(keylists_columns) > 0:
        path_rows = sorted(current_behaviour_state_table[keylists_columns[0]]['behaviour'].keys())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(20.0,7.5)

        # https://matplotlib.org/examples/color/colormaps_reference.html
        colors = plt.cm.Paired(np.linspace(0, 1, len(path_rows)))

        n_rows = len(surprisal_stats)

        index = np.arange(len(keylists_columns)) + 0.4
        bar_width = 0.4
        opacity=0.4
        spare_width = (1 - bar_width*2)/2

        ax.set_xlim(-spare_width,len(index)-spare_width)
        ax.set_ylabel("Surprisal Distribution")
        xTickMarks = keylists_columns

        ax.set_xticks(index+bar_width)

        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=0, fontsize=40)
        ax.yaxis.grid()

        y_offset = np.zeros(len(keylists_columns))

        # Plot bars and create text labels for the table
        cell_text = []
        for row in range(n_rows):
            plt.bar(index, np.minimum(surprisal_stats[row],500), bar_width, bottom=y_offset, color=colors[row])
            y_offset = y_offset + np.minimum(surprisal_stats[row],500)
            cell_text.append(['%.1f' % x for x in surprisal_stats[row]])

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=path_rows,
                              rowColours=colors,
                              alpha=opacity,
                              colLabels=keylists_columns,
                              loc='bottom')
        the_table.scale(1,2.5)
        the_table.auto_set_font_size(value=False)

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)
        ax.xaxis.labelpad = 260
        ax.yaxis.labelpad = 20

        plt.xticks([])
        plt.title('Distribution Of Actions By Behaviour Type')
    return plt

```

## Graph the behaviour groups and the count of members


```python
graph_user_distribution_by_behaviour_id(behaviour_group_table, behaviour_average_table, user_transition_score)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_39_1.png)


## Graph the distribution of by action, weighted by their surprisal score

How to read the graph: 
* Compare the previous graph and this one
* Groups with high number of users and a low surprisal distribution should mainly represent normal users. 
* The different colors should also help to identify which action was mainly used by these users

How to spot outliers?
* Look for groups with a low number of users and a high surprisal distribution.
* The bottom table with the list of actions and scores can also be used to understand the bar above. Low values represent normal actions, high values represents outliers.
* With the distribution of test users used in this notebook, bots should be the main offenders.

Note: The graph below is probably not the best long term solution to graph behaviour groups since it will tend to be crowded with time. It's just for the prouf of concept.


```python
graph_surprisal_distribution_by_action(behaviour_group_table, behaviour_average_table, user_transition_score)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_41_1.png)


# 4.1 Investigate the behaviour groups to understand the behaviour of the members

The following cell extracts the cound of actions for each member of a designated group.



```python
behaviour_group_to_analyse = 43

df_investigate = day1_data.loc[day1_data['user'].isin(behaviour_group_table[behaviour_group_to_analyse])]
df_investigate.groupby(['user', 'status', 'path']).size().unstack(fill_value=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>bank_modify</th>
      <th>end</th>
      <th>login</th>
      <th>sell_item</th>
      <th>update_address</th>
      <th>update_email</th>
      <th>view_item</th>
      <th>view_profile</th>
    </tr>
    <tr>
      <th>user</th>
      <th>status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">merchant1098</th>
      <th>fail</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>success</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# What next: Repeating the analysis on following days

Now that all functions are defined, we can collect logs on subsequent days and classify the users with the previously identified behaviour groups.

If users change their behaviour, they will be reclassified. Inactive users will stay where they are.

Except the "merge_user_transition_score" function, all functions called were previously used. I simply adjusted them to analyse the logs of a different day, in which 200 new users registered.

Logs are also limited to the daily number of users defined at the beginning of the notebook. This means that some users will come back, and some won't. If a user is reclassified, some behaviour group might end up being empty. They will be represented by empty column indexes, but the behaviour average is preserved for further classification if needed. 


```python
# New day
start_time = datetime(2019,1,2,0,0)

# Changing the initial seed to get different results
if random_seed:
    random.seed(random_seed + 1)

# How many new users registered today
number_of_new_users = 200
all_user_lists = generate_userlist(all_user_lists, number_of_new_users)

# Select which users will login today
todays_user_lists = random.sample(all_user_lists, number_of_daily_users)

print('Number of active users today:', len(todays_user_lists), todays_user_lists[:5])

if random_seed:
    random.seed(random_seed + 1)
# Generate the logs for the day for the active users
day2_logs = generate_logs(todays_user_lists, start_time)

print(len(day2_logs), 'logs events generated for', len(todays_user_lists), 'users')

# Prepare the data for analysis, by converting them in a pandas dataframe
day2_data = transform_logs_to_pandas(day2_logs)
```

    Number of active users today: 3000 ['merchant', 'buyer', 'merchant', 'buyer', 'merchant']
    81832 logs events generated for 3000 users



```python
# Calculate the user transition score based on the previous day surprisal lookup table. 
# We could have recompile is here as well, but this is to be consistant with the suggestion to compare users against
# a previously calculated transition table.
user_transition_score_day2 = get_user_transition_score(day2_data, transition_surprisal, 'user', 'path', 'success')
```

## Merge the previously calculated transition scores with the new ones
The next function takes the new transition scores calculated for the second day, and add the values from day 1 for the missing users. This is done to allow us to compare new profiles with previous users when behaviour groups averages are reclaculated.


```python
def merge_user_transition_score(original, newtransitions):        
    for key in original.keys():
        if key not in newtransitions.keys():
            newtransitions[key] = original[key]

    return newtransitions

user_transition_score_merged = merge_user_transition_score(user_transition_score, user_transition_score_day2)
len(user_transition_score_merged)
```




    4546



## Optional (but useful): Identifying unclassified outliers

Same as done on day 1, we can identify outliers just by listing the ones with 2 standard deviations and more.


```python
cumulative_score = [[v,sum(user_transition_score_merged[v].values())] for v in [k for k in list(user_transition_score_merged.keys())]]

df_cumulative_score = pd.DataFrame(cumulative_score, columns=['user', 'surprisal'])

avg = df_cumulative_score['surprisal'].mean()
std = df_cumulative_score['surprisal'].std()
df_cumulative_score['z'] = (df_cumulative_score['surprisal'] - avg) / std

df_cumulative_score.loc[df_cumulative_score['z'] >= 2].sort_values(by=['surprisal'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>surprisal</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>766</th>
      <td>bot240</td>
      <td>106017.762003</td>
      <td>53.350231</td>
    </tr>
    <tr>
      <th>3735</th>
      <td>bot1392</td>
      <td>34192.771972</td>
      <td>17.134004</td>
    </tr>
    <tr>
      <th>4229</th>
      <td>bot831</td>
      <td>29638.440077</td>
      <td>14.837579</td>
    </tr>
    <tr>
      <th>143</th>
      <td>bot2780</td>
      <td>27103.848929</td>
      <td>13.559565</td>
    </tr>
    <tr>
      <th>559</th>
      <td>bot477</td>
      <td>25422.745920</td>
      <td>12.711905</td>
    </tr>
    <tr>
      <th>509</th>
      <td>bot1692</td>
      <td>24911.342416</td>
      <td>12.454041</td>
    </tr>
    <tr>
      <th>3463</th>
      <td>bot524</td>
      <td>22653.550349</td>
      <td>11.315597</td>
    </tr>
    <tr>
      <th>2449</th>
      <td>bot183</td>
      <td>20477.365225</td>
      <td>10.218302</td>
    </tr>
    <tr>
      <th>522</th>
      <td>bot1450</td>
      <td>18866.988233</td>
      <td>9.406303</td>
    </tr>
    <tr>
      <th>4010</th>
      <td>bot2148</td>
      <td>18399.108431</td>
      <td>9.170385</td>
    </tr>
    <tr>
      <th>3268</th>
      <td>bot202</td>
      <td>17577.598547</td>
      <td>8.756156</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>bot1387</td>
      <td>17495.991605</td>
      <td>8.715007</td>
    </tr>
    <tr>
      <th>969</th>
      <td>bot2988</td>
      <td>13203.466447</td>
      <td>6.550593</td>
    </tr>
    <tr>
      <th>3446</th>
      <td>bot194</td>
      <td>8600.834908</td>
      <td>4.229813</td>
    </tr>
    <tr>
      <th>3159</th>
      <td>bot1972</td>
      <td>8094.871867</td>
      <td>3.974692</td>
    </tr>
    <tr>
      <th>3255</th>
      <td>bot691</td>
      <td>7153.671801</td>
      <td>3.500112</td>
    </tr>
    <tr>
      <th>3038</th>
      <td>bot1183</td>
      <td>6756.518015</td>
      <td>3.299856</td>
    </tr>
    <tr>
      <th>791</th>
      <td>bot19</td>
      <td>5396.402313</td>
      <td>2.614046</td>
    </tr>
  </tbody>
</table>
</div>




```python
if random_seed:
    random.seed(random_seed)

unclassified_user_lists = random.sample(list(user_transition_score_merged.keys()), min(len(todays_user_lists), len(list(user_transition_score_merged.keys()))))
print(len(unclassified_user_lists))
```

    3000



```python
%%time

# Classifying observed users
while len(unclassified_user_lists):
    classify_users_in_list(unclassified_user_lists, behaviour_group_table, behaviour_average_table, user_transition_score_merged, maxlimit)
```

    CPU times: user 25.1 s, sys: 105 ms, total: 25.2 s
    Wall time: 25.4 s



```python
# Cheat cell: list all groups with the distribution of their users, by real type

for k in sorted(behaviour_group_table.keys()):
    type_average = np.mean([sum(user_transition_score_merged[x].values()) for x in behaviour_group_table[k]])
    print(k, type_average, len(behaviour_group_table[k]), cheat_lookup_all_users(behaviour_group_table[k]))
```

    0 137.085321251 1262 {'merchant': 1262}
    1 102.285467702 569 {'buyer': 569}
    2 157.054044738 345 {'buyer': 345}
    3 92.0571943776 670 {'buyer': 670}
    4 133.096646896 55 {'buyer': 55}
    5 133.351300167 284 {'merchant': 284}
    6 75.3384737556 1 {'buyer': 1}
    7 181.37947633 4 {'buyer': 3, 'fraudster': 1}
    8 244.894155926 2 {'buyer': 2}
    9 249.65153609 1 {'buyer': 1}
    10 75.7149724417 2 {'merchant': 2}
    11 34192.7719717 1 {'bot': 1}
    12 193.345571213 5 {'merchant': 5}
    13 57.4597311728 7 {'buyer': 7}
    15 232.815195288 7 {'merchant': 7}
    16 82.9709666099 118 {'fraudster': 2, 'attacker': 6, 'buyer': 110}
    17 165.136035105 7 {'spammer': 7}
    18 122.43986214 1 {'buyer': 1}
    19 198.702615172 1 {'merchant': 1}
    20 36.7870271328 7 {'attacker': 4, 'fraudster': 3}
    21 175.210140107 5 {'spammer': 5}
    22 289.152078462 3 {'merchant': 3}
    23 144.355716471 2 {'merchant': 2}
    24 71.0088488556 2 {'fraudster': 2}
    25 28.3528775122 2 {'buyer': 2}
    27 127.938838022 3 {'merchant': 3}
    28 410.621363334 3 {'buyer': 3}
    31 133.241830943 430 {'merchant': 429, 'buyer': 1}
    32 181.8264926 1 {'buyer': 1}
    33 299.439673576 1 {'merchant': 1}
    34 66.8601036056 7 {'attacker': 4, 'fraudster': 3}
    40 154.649574337 4 {'fraudster': 2, 'buyer': 2}
    41 113.699276303 1 {'spammer': 1}
    42 263.772688095 2 {'buyer': 2}
    44 174.878220167 1 {'attacker': 1}
    45 17495.9916045 1 {'bot': 1}
    46 11.5454471817 8 {'buyer': 4, 'merchant': 3, 'attacker': 1}
    47 52.0725412204 2 {'fraudster': 2}
    49 25422.7459202 1 {'bot': 1}
    50 92.3359446749 3 {'buyer': 3}
    51 82.7867919602 95 {'buyer': 95}
    52 163.383394847 7 {'merchant': 7}
    53 18866.9882329 1 {'bot': 1}
    54 106017.762003 1 {'bot': 1}
    55 8600.83490844 1 {'bot': 1}
    56 17577.5985467 1 {'bot': 1}
    57 5396.40231273 1 {'bot': 1}
    58 22653.5503494 1 {'bot': 1}
    59 18399.1084312 1 {'bot': 1}
    60 40.299426061 3 {'merchant': 3}
    61 246.976718812 2 {'buyer': 2}
    62 158.832069554 67 {'buyer': 67}
    63 41.9697630618 10 {'buyer': 10}
    64 20477.365225 1 {'bot': 1}
    65 13203.4664466 1 {'bot': 1}
    66 114.302719058 1 {'buyer': 1}
    67 1272.53150194 1 {'bot': 1}
    68 82.9171337506 1 {'merchant': 1}
    69 6756.51801549 1 {'bot': 1}
    70 69.8039331706 2 {'merchant': 2}
    71 34.1190791622 4 {'buyer': 4}
    72 207.014807237 2 {'buyer': 2}
    73 8094.87186701 1 {'bot': 1}
    74 24911.342416 1 {'bot': 1}
    75 189.837044888 1 {'buyer': 1}
    76 3443.27616355 1 {'bot': 1}
    77 7153.6718007 1 {'bot': 1}
    78 143.840022642 2 {'merchant': 2}
    79 20.5605338383 2 {'buyer': 2}
    80 72.293641481 1 {'spammer': 1}
    81 1092.99622918 1 {'bot': 1}
    82 29638.4400773 1 {'bot': 1}
    83 188.306963181 1 {'merchant': 1}
    84 27.3983706284 1 {'fraudster': 1}


## Graph the behaviour groups and the count of members


```python
graph_user_distribution_by_behaviour_id(behaviour_group_table, behaviour_average_table, user_transition_score_merged)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_55_1.png)


## Graph the distribution of by action, weighted by their surprisal score


```python
graph_surprisal_distribution_by_action(behaviour_group_table, behaviour_average_table, user_transition_score_merged)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_57_1.png)


## Investigate the behaviour groups to understand the behaviour of the members


```python
behaviour_group_to_analyse = 84

df_investigate = day2_data.loc[day2_data['user'].isin(behaviour_group_table[behaviour_group_to_analyse])]
df_investigate.groupby(['user', 'status', 'path']).size().unstack(fill_value=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>end</th>
      <th>login</th>
      <th>logout</th>
    </tr>
    <tr>
      <th>user</th>
      <th>status</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fraudster75</th>
      <th>success</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


