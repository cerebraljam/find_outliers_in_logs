
# How to group users and identify outliers in a log stream, based on behaviour


## What is demonstrated in this notebook

In this notebook I demonstrate, from a (generated) stream of logs, how to group users by behaviour, and how to highlight which behaviour group an analyst should focus his attention on. 

This method is good when monitoring a stable and closed system, with a finite number of possible actions, and known definition of success and failure.

Probabilities are based on a "Subject (unique key), Action (ex.: login), Outcome (success/fail)" discrete format. Continuous Action variables (like numerical values) would need to be reduced to discrete output to fit this model.

## Why is this a hard problem?

What is a normal action? Which one is suspect? The answer is "it depends". 

To group individual together, we need a way to classify each action taken over time, and somehow group them together.

A clustering function like KMean requires us to know how many cluster we have, and we need mesurable features to group them together. Normal clustering functions lose in efficiency the more we add features.

Machine learning techniques are also great to identify patterns over a large number of repeating features, but struggle to handle unusual events especially when data is skewed by a high volume of "normal" actions. This is why when we do text analysis, we normally remove common [English stop words](https://www.ranks.nl/stopwords). But what if these stop words were necessary to identfy an outlier? How can we keep them without debalancing everything?

One other hypothesis would be to use Hidden Markov Chains, but for HNN, we need classified data. Sequences also need to be somehow of a consistent length, but users are most likely not to follow a straight pattern from A to B, which complicates the creation of HMM.

Another approach would be to use rules. Classically, Intrusion detection systems recognize a set of patterns, and uses thresholds to know when to raise a flag, generating a large number of false positive. A rule based system is also unlikely to be able to detect the usage of normal functions in an abusive way.

In the end, we need each features from all possible events to be classified and scored somehow, and we need a way to group these scores together.

## Approach taken

### What this notebook is the following

* Step 1: We calculate the probability of each sequences of actions being taken by all users over a defined period of time (ex.: 1 day). From these probabilities, we create a lookup probability table for each possible sequence.
* Step 2: For each users, we sum the probability (using logarithm base 2) for each action taken. Depending on the likelyhood of each action, this tells us how much new information we learned from the actions taken by a user, defining a action profile for each user.
* Step 3: Using the profiles created on Step 2, we group them together by calculating how much information is gained if we compare each actions together. If the information gained is low, then we know that both users had a similar action pattern. If the information gained is high, then they have different behaviour.
* Step 4: With all users grouped together, we can quickly analyse a group of user and label that group with a usage profile. The way these profiles are caculated here should keep them stable over time. If new behaviours are observed, new groups will be created. An analyst then only have to pay attention to the new group and groups identified as malicious when monitoring a system.

### Caviats

* For Step 1, this can be done on a live stream of logs, but for systems generating a large quantify of logs, this could result in integer overflow if we are dealing with too many events. Using rolling windows (ex.: calculate for a period, then use these probabilities distribution to analyse the next period, while creating a new distribution for the next period) might be a better solution.
* For Step 2, this works better when there is a clear start and end of sequence for a user. As it is coded right now, scoring a partial session would end up creating a large quantity of behaviour groups, which would be noisy.
* For Step 3, the "distance" (which isn't) between each users is somewhat arbitrary selected. Litmits should be adjusted depending on the number of events on the system. On small simulations, larger values (5-10) worked well. However, if the quantity of events is really high, a smaller distance (1-3) might be necessary. Using too strict limits will generate a large number of groups to represent all possible behaviours, but if we use a larger maximum limit (ex.: 10), then we increase the likelihood of having misclassifications and group outlier with normal users.
* For Step 4: None here :-D This is where I would start configuring rules or send data to a machine learning model, because users are now classified in limited groups.

## The theory behind the approach

Formulas are inspired from Information Theory, and they are normally used for signal analysis and compression. 
* Step 2 consist of using Surprisal: by calculating the entropy of each action, we can assign a score on each action, representing how surprised we should be when we see an action happening.
* Step 3 uses the Kullback–Leibler divergence formula. This normally tells us how much information is gained by comparing two probabilities together. However, this formula isn't symetric. By calculating both possibilities, we can get an idea how close two profiles are, and use this to group similar behaviours together.

* [Information Content](https://en.wikipedia.org/wiki/Information_content)
* [Surprisal Analysis](https://en.wikipedia.org/wiki/Surprisal_analysis)
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
* [Example of Surprisal](http://www.umsl.edu/~fraundorfp/egsurpri.html)

Example of how to calculate surprisal:
* There is 1/6 chance to roll a 6 on a d6 die, gives us -log2(1/6) ~= 2.4849 bits of information. If you are playing D&D, you will be happily surprised.
* Not rolling a 6 on a die represents -log2(5/6) ~= 0.263 bits. So rolling anything but a 6 in that same D&D game will most likely leave you dissapointed. 
* The chance of rolling 10 x 6 in a row on a die is 1/60466176 (10 * -log2(1/6) ~= 25.85 bits). This is highly unlikely.

So later when we look at the surprisal score of a user and we see 51550 (bits), this tells us that the actions taken by this user are really, really unlikely.

This also tells us that if the total surprisal score of the actions of a user is low, then we can safely assume that the actions taken by that user are normal. But by repeating a large number of normal actions together, the score will pile up and single out the user.

Surprisal are calculated using Log base 2, which has the nice particularity to be additive instead of being multiplied like probabilities, keeping numbers relatively small and easy to manipulate.

## Demonstration

This is what this notebook intend to demonstrate: Using Surprisal analysis, we will assign a score to actions, and by adding up the score of each action, identify series of actions that are unlikely to occur, then group users by their action profile.


# Step 0: We need logs

What is explained in this notebook can be applied to real logs, but for the experimentation, I generate logs for simulated users based on a determined probability distribution.

## Caviat: I use some cheats to demonstrate how the notebook works...

These cheats are not necessary to make this technique work and are clearly identified when used. I did use one of these cheat at first to identify the most efficient Kullback-Leibler divergence limit to use in Step 3. 


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
import timeit

chrono_start = timeit.default_timer()

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
number_of_daily_users = 2000 # The more we add users, the longer the notebook takes to run.
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

print(len(todays_user_lists), 'users in the database.')
print('Type of the 15 firsts:', todays_user_lists[:15])
```

    2000 users in the database.
    Type of the 15 firsts: ['buyer', 'merchant', 'merchant', 'buyer', 'buyer', 'buyer', 'buyer', 'buyer', 'merchant', 'merchant', 'merchant', 'merchant', 'buyer', 'buyer', 'buyer']


# Step 0.2: Generating Logs For Day 1

Note: The more users we have, the more log events will be generated. 

The probability distribution of each user ensures that they will start with a defined action, crawl the site following a defined pattern, and logout eventually, until the end of the day.


```python
%%time
if random_seed:
    random.seed(random_seed)

start_time = datetime(2019,1,1,0,0)
day1_logs = generate_logs(todays_user_lists, start_time)

print(len(day1_logs), 'log events generated for', len(todays_user_lists), 'users')
```

    40170 log events generated for 2000 users
    CPU times: user 7.44 s, sys: 52.8 ms, total: 7.49 s
    Wall time: 7.61 s


## Transforming the logs in a pandas dataframe (for this notebook...)

The transition surprisal lookup table used in this notebook calculates scores based on the movements of the users between each actions. For example:

* login -> view_items (success) will result in a low surpisal value
* login -> buy_item (success) never happened. If this sequence happen, this should be a huge red flag.

Note: I skipped one feature: the status (success/failure) of the previous path isn't taken into consideration. Adding it would definitely make the calculations more sensitive to anomalies, but with the cost of an increase on complexity.


```python
def transform_logs_to_pandas(logs):
    data = pd.DataFrame(np.array(logs), columns=['time', 'user', 'path', 'status', 'uidx', 'realtype'])
    
    data['prev_path'] = data.groupby(['user'])['path'].shift(1)
    data['prev_path'] = data['prev_path'].fillna("")
    return data
    
day1_data = transform_logs_to_pandas(day1_logs)


# Example of failed actions in the logs. uidx and realtype are "cheat" columns, and not necessary in a real case usage.
print(day1_data.loc[(day1_data['path'] == 'login') & (day1_data['status'] == 'fail')].head())
```

                        time          user   path status  uidx  realtype prev_path
    558  2019-01-01 00:46:39  merchant1316  login   fail  1316  merchant          
    777  2019-01-01 00:59:12   attacker652  login   fail   652  attacker          
    778  2019-01-01 00:59:15   attacker652  login   fail   652  attacker     login
    779  2019-01-01 00:59:19   attacker652  login   fail   652  attacker     login
    834  2019-01-01 01:03:14     buyer1472  login   fail  1472     buyer          


## Step 1 : Generate the transition lookup table

**This is where the magic trick happens.**

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

The surprisal value is directly related to the likelihood of an actions happening. If an actions is observed successfully few million times, then the successful surprisal value will be really low. However, the failure surprisal will be much higher if an action/status never or rarely happens.


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
                'success': len(ds),  # Not used later one, just interesting for debugging
                'fail': len(df),  # Not used later one.
                'ssurprisal': log(1/(dsuccess / denum),2), # Magic!!
                'fsurprisal': log(1/(dfail / denum),2), # Magic!!
            }
    return surprisal

transition_surprisal = init_transition_surprisal_lookup(day1_data, 'path', 'prev_path', 'status', 'success')
```

The next cell creates a wrapper function to query the transition surprisal lookup table. If the sequence was previously observed, it will return it, and will return the probability of 1 over the total number of actions taken (unlikely) for unobserved events, making them stand out.


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




    {'fail': 0,
     'fsurprisal': 11.213711798105674,
     'ssurprisal': 6.965784284662088,
     'success': 19}



# Step 2: For each users, sum the probability for all actions taken.

The following function takes the dataframe with all the logs for a day, and returns the sum of the surprisal values, conditional to a success or a failure, for all actions taken by a user.


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

## Optional (but useful): Identify who are the main offenders

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
      <th>686</th>
      <td>bot272</td>
      <td>51550.564081</td>
      <td>39.684129</td>
    </tr>
    <tr>
      <th>623</th>
      <td>bot1295</td>
      <td>25659.755591</td>
      <td>19.690524</td>
    </tr>
    <tr>
      <th>189</th>
      <td>bot1323</td>
      <td>4788.884029</td>
      <td>3.573455</td>
    </tr>
    <tr>
      <th>832</th>
      <td>bot1292</td>
      <td>3741.640951</td>
      <td>2.764745</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>bot321</td>
      <td>3121.263258</td>
      <td>2.285672</td>
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
      <th>1996</th>
      <td>buyer528</td>
      <td>16.301175</td>
      <td>-0.112066</td>
    </tr>
    <tr>
      <th>532</th>
      <td>fraudster152</td>
      <td>16.046740</td>
      <td>-0.112263</td>
    </tr>
    <tr>
      <th>1623</th>
      <td>fraudster1221</td>
      <td>16.046740</td>
      <td>-0.112263</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>attacker1699</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>458</th>
      <td>merchant1421</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>5</th>
      <td>buyer282</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>391</th>
      <td>fraudster184</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>1833</th>
      <td>attacker1512</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>249</th>
      <td>buyer1712</td>
      <td>10.959278</td>
      <td>-0.116191</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>buyer1367</td>
      <td>5.087463</td>
      <td>-0.120726</td>
    </tr>
  </tbody>
</table>
</div>



## Cheat cell: Identify the ideal "distance" to increase the True Positive rate, and reduce the False Positive Rate

Without using this cheat, we need to guess the ideal Kullback-Leibler divergence limit. 

* From experimentation, this should be between 4 and 8. 9 and more did tend to cause a higher False Positive rate.
* Values too small are likely to reduces the number of false positives, but will end up create more behaviour profiles
* Values too big will result in a smaller number of groups, and more misclassification



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

    best 2 {True: {True: 0.937984496124031, False: 0.12451577199778639}, False: {True: 0.06201550387596899, False: 0.8754842280022136}} 0.937984496124031 0.12451577199778639 0.5312501340609087 2
    best 4 {True: {True: 0.9398496240601504, False: 0.12257348863006101}, False: {True: 0.06015037593984962, False: 0.877426511369939}} 0.9398496240601504 0.12257348863006101 0.5312115563451056 4
    limit 4
    count True {True: 120, False: 8}
    count False {True: 226, False: 1582}
    percent True {True: 0.9375, False: 0.125}
    percent False {True: 0.0625, False: 0.875}



```python
# Bypassing the cheat function and forcing the maximum Kullback-Leibler divergence limit.
maxlimit = 5
```

# Step 3: Grouping users by their behaviour

The second magic trick is in the compare_profiles() function


```python
# Calculating the "distance" between profiles requires to convert log base 2 back to probabilities, 
# which sometimes causes overflow when profiles are too far apart. We can safely ignore these errors
np.seterr(divide='ignore', invalid='ignore', over='ignore') 

# If new actions are observed and are missing, we need to 1) ignore them, or 2) add them for the calculation
# This function ensure that both profiles has the same keys, and asign 0 when missing
# Profiles and keys are returned in order of key names
def align_profiles(profile1, profile2):
    if profile1.keys() != profile2.keys():
        for k in profile1.keys():
            if k not in profile2.keys():
                profile2[k] = 0#profile1[k]
        for k in profile2.keys():
            if k not in profile1.keys():
                profile1[k] = 0#profile2[k]
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
    kldp = (qx * np.log2(p)).sum()    
    kldq = (px * np.log2(q)).sum()
    
    # kldp/kldq == Kullback-Leibler Divergence for p and q
    t = (kldp < limit and kldp >= -limit) and (kldq < limit and kldq >= -limit)
    
    return {'test': t, 'kldp': kldp, 'kldq': kldq}

# Example of usage
test_users = random.sample(user_transition_score.keys(),2)
print(test_users[0], user_transition_score[test_users[0]])
print(test_users[1], user_transition_score[test_users[1]])
compare_profiles(user_transition_score[test_users[0]], user_transition_score[test_users[1]], maxlimit)
```

    merchant857 {'login': 5.08746284125034, 'view_item': 14.71112919630353, 'comment': 0, 'buy_item': 0, 'logout': 0, 'end': 10.959277505720502, 'sell_item': 10.349281228817453, 'home': 0, 'bank_modify': 0, 'view_profile': 0, 'password_reset': 0, 'payment_modify': 0, 'update_address': 0, 'update_email': 0}
    merchant946 {'login': 5.08746284125034, 'view_item': 129.07211050631952, 'comment': 0, 'buy_item': 0, 'logout': 0, 'end': 10.959277505720502, 'sell_item': 102.84895609839981, 'home': 0, 'bank_modify': 0, 'view_profile': 0, 'password_reset': 0, 'payment_modify': 0, 'update_address': 0, 'update_email': 0}





    {'kldp': -1.0127900003221136e-29, 'kldq': 0.0751718455254234, 'test': True}




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




    {'a': 1.5, 'b': 1.75, 'c': 3.25, 'd': 1.0}



Once classified, this function will add to the identified class.
* If the target class does not exist yet (the classification function returns an 
* Inexistant class the candidate didn't match any existing profile)
* Then the function will add the candidate to that class.


```python
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
    
    for be, be_average in behaviour_type_average.items():
        result = compare_profiles(score[candidate_name], be_average, limit)
            
        if result['test'] == True:
            potential_matching_type[be] = result['kldp']**2 + result['kldq']**2

    if len(potential_matching_type.keys()) == 0:
        new_class_name = max(0,len(list(behaviour_type_average.values())))
        return new_class_name
    else:
        return min(potential_matching_type, key=potential_matching_type.get)

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
Then we get the list of observed users, and we create a list of unclassified users to classify.


```python
if random_seed:
    random.seed(random_seed)

behaviour_group_table = {}
behaviour_average_table = {}
print(len(list(user_transition_score.keys())))
unclassified_user_lists = random.sample(list(user_transition_score.keys()), min(len(todays_user_lists), len(list(user_transition_score.keys()))))
```

    1999


The next cell is heavy to run. This is where we go through the list of unclassified users and we add them in the proper group.

Note: *behaviour_group_table* and *behaviour_average_table* are modified by the subfunctions


```python
%%time
while len(unclassified_user_lists):
    classify_users_in_list(unclassified_user_lists, behaviour_group_table, behaviour_average_table, user_transition_score, maxlimit)
```

    CPU times: user 8 s, sys: 116 ms, total: 8.11 s
    Wall time: 8.94 s


## Cheat cell: List all behaviour groups and the number of user by their real type

This cell is only to demonstrate that the previous functions mostly group users by their good real time.
There are some misclassifications, this could be caused by the randomness of the actions taken by the generated users. 

Also, an attacker's action could look exactly like a real user. In these cases, we will most probably won't be able to identify that that user was an attacker, unless other actions were taken that would separate the behaviour profile from a normal user.


```python
for k in behaviour_group_table.keys():
        type_average = np.mean([sum(user_transition_score[x].values()) for x in behaviour_group_table[k]])
        print(k, type_average, len(behaviour_group_table[k]), cheat_lookup_all_users(behaviour_group_table[k]))
```

    0 133.81558114 528 {'merchant': 528}
    1 89.4563711812 515 {'buyer': 515}
    2 122.584854049 476 {'merchant': 476}
    3 157.83356633 165 {'buyer': 165}
    4 145.142789844 56 {'buyer': 56}
    5 82.4124937823 201 {'buyer': 201}
    6 17.9298542566 3 {'fraudster': 2, 'attacker': 1}
    7 10.9592775057 6 {'attacker': 2, 'buyer': 2, 'fraudster': 1, 'merchant': 1}
    8 25659.7555907 1 {'bot': 1}
    9 39.0774246392 7 {'buyer': 7}
    10 28.3358997395 2 {'buyer': 1, 'merchant': 1}
    11 158.391936923 2 {'merchant': 2}
    12 140.563175357 4 {'spammer': 4}
    13 5.08746284125 1 {'buyer': 1}
    14 31.5748267938 2 {'buyer': 2}
    15 94.2536470715 4 {'attacker': 4}
    16 167.587890011 3 {'fraudster': 1, 'buyer': 2}
    17 206.475582662 1 {'merchant': 1}
    18 168.933119571 1 {'spammer': 1}
    19 16.3011746394 1 {'buyer': 1}
    20 77.5364574116 1 {'buyer': 1}
    21 228.95766815 2 {'merchant': 2}
    22 373.794311498 1 {'spammer': 1}
    23 190.819622188 2 {'buyer': 1, 'fraudster': 1}
    24 111.698395602 2 {'merchant': 2}
    25 48.4507278436 1 {'merchant': 1}
    26 3121.26325764 1 {'bot': 1}
    27 26.3228647522 1 {'fraudster': 1}
    28 471.164325798 1 {'merchant': 1}
    29 51550.5640814 1 {'bot': 1}
    30 181.700719883 1 {'buyer': 1}
    31 63.1502967645 1 {'buyer': 1}
    32 168.427842172 1 {'buyer': 1}
    33 4788.88402875 1 {'bot': 1}
    34 183.165776658 1 {'merchant': 1}
    35 181.943107272 1 {'buyer': 1}
    36 3741.64095064 1 {'bot': 1}


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




![png](find_outliers_in_logs_files/find_outliers_in_logs_40_1.png)


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




![png](find_outliers_in_logs_files/find_outliers_in_logs_42_1.png)


# 4.1 Investigate the behaviour groups to understand the behaviour of the members

The following cell extracts the count of actions for each member of a designated group.



```python
behaviour_group_to_analyse = 26

df_investigate = day1_data.loc[day1_data['user'].isin(behaviour_group_table[behaviour_group_to_analyse])]
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
      <th>view_item</th>
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
      <th rowspan="2" valign="top">bot321</th>
      <th>fail</th>
      <td>0</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>success</th>
      <td>1</td>
      <td>1</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>



# What next: Repeating the analysis on following days

Now that all functions are defined, we can collect logs on subsequent days and classify the users with the previously identified behaviour groups.

If users change their behaviour, they will be reclassified. Inactive users will stay where they are.

Except the "merge_user_transition_score" function, all functions called were previously used. I simply adjusted them to analyse the logs of a different day, in which 200 new users registered.

Event logs are also limited to the daily number of users defined at the beginning of the notebook. This means that some users will come back, and some won't. If a user is reclassified, some behaviour group might end up being empty. Empty groups will be deleted in *behaviour_group_table* but will remain in *behaviour_average_table* for further classification if needed. 

Empty groups will be represented by empty column indexes in the graph, but the behaviour average is preserved  


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

    Number of active users today: 2000 ['merchant', 'buyer', 'merchant', 'buyer', 'merchant']
    36626 logs events generated for 2000 users



```python
# Calculate the user transition score based on the previous day surprisal lookup table. 
# We could have recompile is here as well, but this is to be consistant with the suggestion to compare users against
# a previously calculated transition table.
user_transition_score_day2 = get_user_transition_score(day2_data, transition_surprisal, 'user', 'path', 'success')
```

## Merge the previously calculated transition scores with the new ones
The next function takes the new user transition scores calculated for the second day, and add the values from day 1 for the missing users. This is done to allow us to compare new profiles with previous users when behaviour groups averages are reclaculated.


```python
def merge_user_transition_score(original, newtransitions):        
    for key in original.keys():
        if key not in newtransitions.keys():
            newtransitions[key] = original[key]

    return newtransitions

user_transition_score_merged = merge_user_transition_score(user_transition_score, user_transition_score_day2)
len(user_transition_score_merged)
```




    3015



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
      <th>2346</th>
      <td>bot272</td>
      <td>51550.564081</td>
      <td>41.683575</td>
    </tr>
    <tr>
      <th>2318</th>
      <td>bot1295</td>
      <td>25659.755591</td>
      <td>20.679847</td>
    </tr>
    <tr>
      <th>204</th>
      <td>bot583</td>
      <td>21476.474817</td>
      <td>17.286191</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>bot1919</td>
      <td>19934.067892</td>
      <td>16.034925</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>bot1125</td>
      <td>16706.965581</td>
      <td>13.416962</td>
    </tr>
    <tr>
      <th>1809</th>
      <td>bot387</td>
      <td>7947.687878</td>
      <td>6.311062</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>bot1141</td>
      <td>6650.017108</td>
      <td>5.258336</td>
    </tr>
    <tr>
      <th>2099</th>
      <td>bot1323</td>
      <td>4788.884029</td>
      <td>3.748506</td>
    </tr>
    <tr>
      <th>2429</th>
      <td>bot1292</td>
      <td>3741.640951</td>
      <td>2.898938</td>
    </tr>
    <tr>
      <th>2941</th>
      <td>bot321</td>
      <td>3121.263258</td>
      <td>2.395661</td>
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

    2000



```python
%%time

# Classifying observed users
while len(unclassified_user_lists):
    classify_users_in_list(unclassified_user_lists, behaviour_group_table, behaviour_average_table, user_transition_score_merged, maxlimit)
```

    CPU times: user 12.6 s, sys: 131 ms, total: 12.7 s
    Wall time: 13.2 s



```python
# Cheat cell: list all groups with the distribution of their users, by real type

for k in sorted(behaviour_group_table.keys()):
    type_average = np.mean([sum(user_transition_score_merged[x].values()) for x in behaviour_group_table[k]])
    print(k, type_average, len(behaviour_group_table[k]), cheat_lookup_all_users(behaviour_group_table[k]))
```

    0 137.038212747 692 {'merchant': 692}
    1 92.0019026784 408 {'buyer': 408}
    2 120.024466698 294 {'merchant': 294}
    3 154.026313657 212 {'buyer': 212}
    4 137.186219547 36 {'buyer': 36}
    5 92.0583845405 394 {'buyer': 394}
    6 34.6424902045 6 {'fraudster': 2, 'attacker': 4}
    7 10.9592775057 9 {'attacker': 3, 'buyer': 2, 'fraudster': 1, 'merchant': 3}
    8 25659.7555907 1 {'bot': 1}
    9 40.872344293 9 {'buyer': 9}
    11 176.840289849 3 {'merchant': 3}
    12 161.193528137 5 {'spammer': 5}
    13 5.08746284125 2 {'buyer': 1, 'merchant': 1}
    14 29.9569363005 4 {'buyer': 4}
    15 95.3283777732 4 {'attacker': 4}
    16 155.05215635 5 {'fraudster': 2, 'buyer': 3}
    17 143.75722284 2 {'merchant': 2}
    18 168.933119571 1 {'spammer': 1}
    19 16.3011746394 1 {'buyer': 1}
    20 77.5364574116 1 {'buyer': 1}
    21 308.494259733 2 {'merchant': 2}
    22 216.923595102 2 {'spammer': 2}
    23 153.70212355 3 {'buyer': 2, 'fraudster': 1}
    24 110.012629923 3 {'merchant': 3}
    25 10.9592775057 1 {'merchant': 1}
    26 3121.26325764 1 {'bot': 1}
    27 26.3228647522 1 {'fraudster': 1}
    28 57.0748136453 1 {'merchant': 1}
    30 216.428184269 2 {'buyer': 2}
    31 63.1502967645 1 {'buyer': 1}
    32 168.427842172 1 {'buyer': 1}
    33 4788.88402875 1 {'bot': 1}
    34 183.165776658 1 {'merchant': 1}
    35 181.943107272 1 {'buyer': 1}
    36 3741.64095064 1 {'bot': 1}
    37 78.0723068109 169 {'buyer': 169}
    38 130.914991089 52 {'buyer': 52}
    39 7947.68787846 1 {'bot': 1}
    40 116.385079767 351 {'merchant': 351}
    41 39.2988555519 4 {'merchant': 4}
    42 182.112616571 1 {'buyer': 1}
    43 51550.5640814 1 {'bot': 1}
    44 182.162236809 1 {'merchant': 1}
    45 21476.4748168 1 {'bot': 1}
    46 19934.0678919 1 {'bot': 1}
    47 16706.9655806 1 {'bot': 1}


## Graph the behaviour groups and the count of members


```python
graph_user_distribution_by_behaviour_id(behaviour_group_table, behaviour_average_table, user_transition_score_merged)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_56_1.png)


## Graph the distribution of by action, weighted by their surprisal score


```python
graph_surprisal_distribution_by_action(behaviour_group_table, behaviour_average_table, user_transition_score_merged)
```




    <module 'matplotlib.pyplot' from '/Users/simon/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py'>




![png](find_outliers_in_logs_files/find_outliers_in_logs_58_1.png)


## Investigate the behaviour groups to understand the behaviour of the members


```python
behaviour_group_to_analyse = 39

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
      <th>view_item</th>
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
      <th rowspan="2" valign="top">bot387</th>
      <th>fail</th>
      <td>0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>success</th>
      <td>1</td>
      <td>1</td>
      <td>1364</td>
    </tr>
  </tbody>
</table>
</div>



# Step 5: Identifying the behaviour group without classifying the users

The method demonstrated previously is good when user sessions are completed and to classify them after the fact, but what if we wanted to classify them as they were using the site?

This would require few major changes in the code:
* Statistics on all actions would need to be calculated on an ongoing basis
* The surprisal value of each action (positive and negative) could be calculated when we queried the system to classify the user
* The classification function would need to return the class without updating the behaviour group and average table, or return "unknown" when the user doesn't match any group. This could be an issue if we only look at the class of user since we wouldn't be able to tell if it was because he didn't perform enough actions to be classified, or because he is currently hammering the site with a new unobserved attack, but if we couple this "unknown" class with the sum of all surprisal values for the actions taken by the user, this would allow the analysts to have a good idea what he is doing anyway.

This section is not presenting a full implementattion of these change, but will demonstrate how a user could be classified without updating the model

## 5.1: Generating the logs for Day 3

This next cell is just like the day 2 initialisation cell, and code would need to be modified to handle a stream of logs, but we can still simulate this behaviour without rewriting the whole notebook by going through each lines of the day3_data pandas dataframe as if it was the parsed output of our logs.


```python
# New day
start_time = datetime(2019,1,3,0,0)

# Changing the initial seed to get different results
if random_seed:
    random.seed(random_seed + 2)

# How many new users registered today
number_of_new_users = 200
all_user_lists = generate_userlist(all_user_lists, number_of_new_users)

# Select which users will login today
todays_user_lists = random.sample(all_user_lists, number_of_daily_users)

print('Number of active users today:', len(todays_user_lists), todays_user_lists[:5])

if random_seed:
    random.seed(random_seed + 2)
# Generate the logs for the day for the active users
day3_logs = generate_logs(todays_user_lists, start_time)

print(len(day3_logs), 'logs events generated for', len(todays_user_lists), 'users')

# Prepare the data for analysis, by converting them in a pandas dataframe
day3_data = transform_logs_to_pandas(day3_logs)
```

    Number of active users today: 2000 ['merchant', 'merchant', 'buyer', 'buyer', 'buyer']
    34914 logs events generated for 2000 users


## 5.2: Calculating the user transition surprisal score for each users

This cell is calculating the transition surprisal score for the full day, and would need to be modified to handle a stream of logs.


```python
user_transition_score_day3 = get_user_transition_score(day3_data, transition_surprisal, 'user', 'path', 'success')

user_transition_score_merged = merge_user_transition_score(user_transition_score_merged, user_transition_score_day3)
print('Total number of transition scores:', len(user_transition_score_merged))

```

    Total number of transition scores: 3552



```python
if random_seed:
    random.seed(random_seed + 11)
# sample_users = random.sample(, 20)
listing_of_users = list(day3_data['user'].unique())
for i in range(len(listing_of_users)):
    print(i, listing_of_users[i])
```

    0 buyer495
    1 buyer1391
    2 merchant1934
    3 buyer762
    4 buyer1122
    5 merchant1605
    6 merchant867
    7 merchant1987
    8 buyer1332
    9 merchant1219
    10 merchant1429
    11 buyer712
    12 merchant1305
    13 merchant284
    14 merchant175
    15 merchant189
    16 merchant987
    17 buyer98
    18 buyer853
    19 merchant298
    20 merchant1691
    21 buyer1684
    22 buyer1427
    23 merchant1659
    24 buyer106
    25 buyer934
    26 merchant956
    27 merchant1314
    28 merchant1802
    29 merchant442
    30 buyer1004
    31 buyer12
    32 merchant1071
    33 merchant962
    34 merchant70
    35 buyer1734
    36 buyer308
    37 buyer968
    38 buyer344
    39 buyer572
    40 merchant708
    41 merchant1500
    42 merchant500
    43 merchant485
    44 buyer1611
    45 buyer1169
    46 buyer430
    47 buyer1408
    48 buyer898
    49 buyer1614
    50 buyer889
    51 buyer393
    52 merchant192
    53 merchant389
    54 buyer72
    55 buyer1518
    56 merchant1966
    57 merchant1681
    58 merchant520
    59 buyer200
    60 buyer346
    61 merchant1759
    62 merchant229
    63 buyer224
    64 buyer1829
    65 merchant645
    66 buyer548
    67 buyer583
    68 merchant908
    69 merchant329
    70 buyer1991
    71 buyer1569
    72 merchant1986
    73 merchant1833
    74 buyer288
    75 buyer843
    76 buyer1582
    77 merchant534
    78 buyer1526
    79 merchant1830
    80 merchant443
    81 merchant1129
    82 merchant812
    83 buyer1709
    84 buyer477
    85 merchant779
    86 merchant1556
    87 buyer1194
    88 merchant1064
    89 merchant1768
    90 buyer609
    91 buyer623
    92 buyer385
    93 buyer886
    94 merchant324
    95 merchant902
    96 merchant1246
    97 merchant1347
    98 buyer1079
    99 buyer1199
    100 buyer1351
    101 merchant932
    102 merchant8
    103 merchant342
    104 merchant509
    105 buyer1472
    106 merchant1985
    107 buyer1046
    108 merchant1170
    109 buyer579
    110 buyer619
    111 merchant675
    112 buyer510
    113 buyer1740
    114 buyer279
    115 buyer1306
    116 buyer225
    117 buyer491
    118 merchant871
    119 buyer154
    120 buyer728
    121 buyer1762
    122 attacker38
    123 buyer706
    124 buyer1331
    125 merchant1151
    126 buyer584
    127 merchant1883
    128 buyer1499
    129 merchant1855
    130 buyer681
    131 buyer1949
    132 spammer52
    133 buyer598
    134 bot1607
    135 buyer630
    136 merchant1031
    137 buyer984
    138 buyer1806
    139 buyer404
    140 merchant1454
    141 merchant1045
    142 buyer1686
    143 merchant1128
    144 merchant1292
    145 buyer468
    146 merchant754
    147 merchant273
    148 buyer1260
    149 buyer415
    150 merchant870
    151 buyer637
    152 buyer1216
    153 buyer50
    154 buyer1665
    155 buyer1708
    156 merchant67
    157 merchant1050
    158 buyer1644
    159 merchant343
    160 merchant1598
    161 merchant1040
    162 merchant566
    163 merchant472
    164 merchant1897
    165 buyer1085
    166 merchant816
    167 merchant647
    168 buyer1509
    169 buyer301
    170 buyer1895
    171 merchant564
    172 buyer1662
    173 merchant1713
    174 buyer91
    175 buyer599
    176 merchant1322
    177 buyer1722
    178 buyer1787
    179 buyer313
    180 merchant49
    181 buyer542
    182 merchant1963
    183 buyer1011
    184 buyer1746
    185 buyer1616
    186 buyer207
    187 merchant560
    188 buyer327
    189 buyer1383
    190 spammer700
    191 buyer359
    192 buyer1052
    193 merchant328
    194 buyer1165
    195 buyer575
    196 merchant1121
    197 merchant1610
    198 buyer1542
    199 merchant1489
    200 merchant730
    201 buyer1054
    202 merchant1780
    203 merchant684
    204 buyer1650
    205 merchant694
    206 merchant89
    207 buyer311
    208 buyer1265
    209 merchant1868
    210 merchant1670
    211 merchant113
    212 merchant1340
    213 buyer768
    214 merchant1035
    215 buyer836
    216 buyer321
    217 buyer1258
    218 buyer1941
    219 merchant978
    220 attacker103
    221 merchant117
    222 buyer123
    223 buyer1172
    224 merchant216
    225 merchant894
    226 buyer810
    227 fraudster1414
    228 merchant1839
    229 merchant494
    230 merchant958
    231 merchant1030
    232 merchant1838
    233 merchant1885
    234 merchant663
    235 merchant30
    236 buyer1263
    237 buyer1788
    238 buyer643
    239 buyer1107
    240 buyer752
    241 merchant1393
    242 buyer1676
    243 merchant1585
    244 buyer1775
    245 buyer132
    246 buyer1281
    247 merchant158
    248 merchant314
    249 merchant847
    250 merchant23
    251 merchant1718
    252 buyer1546
    253 buyer139
    254 merchant105
    255 buyer231
    256 bot1076
    257 merchant859
    258 merchant365
    259 merchant326
    260 merchant201
    261 merchant1922
    262 merchant471
    263 buyer563
    264 buyer176
    265 merchant780
    266 merchant1521
    267 merchant1036
    268 buyer918
    269 buyer972
    270 merchant1724
    271 merchant766
    272 buyer596
    273 buyer1145
    274 buyer1917
    275 buyer1912
    276 merchant1534
    277 merchant432
    278 buyer990
    279 buyer1597
    280 buyer1446
    281 merchant1067
    282 merchant939
    283 buyer844
    284 buyer1190
    285 merchant145
    286 merchant1642
    287 buyer473
    288 merchant1171
    289 buyer704
    290 merchant1089
    291 merchant380
    292 merchant669
    293 buyer518
    294 buyer1440
    295 merchant1116
    296 merchant445
    297 buyer639
    298 buyer1326
    299 buyer1191
    300 buyer349
    301 buyer565
    302 buyer778
    303 buyer1679
    304 buyer979
    305 merchant1303
    306 buyer633
    307 buyer1990
    308 merchant621
    309 merchant32
    310 merchant1062
    311 buyer203
    312 buyer373
    313 merchant41
    314 merchant1238
    315 merchant1545
    316 merchant283
    317 merchant1187
    318 buyer1019
    319 merchant1579
    320 buyer551
    321 merchant1339
    322 merchant13
    323 merchant1342
    324 merchant1772
    325 buyer1628
    326 merchant952
    327 buyer1028
    328 merchant1271
    329 merchant1733
    330 buyer502
    331 merchant1300
    332 merchant672
    333 merchant796
    334 buyer677
    335 buyer1824
    336 merchant1450
    337 buyer1017
    338 buyer420
    339 merchant1968
    340 merchant1005
    341 buyer1608
    342 merchant35
    343 merchant1726
    344 buyer971
    345 merchant1063
    346 buyer1807
    347 buyer790
    348 merchant1090
    349 merchant83
    350 buyer1529
    351 buyer335
    352 buyer1143
    353 merchant28
    354 merchant263
    355 merchant1250
    356 buyer86
    357 buyer368
    358 buyer1752
    359 merchant170
    360 merchant350
    361 merchant665
    362 buyer743
    363 buyer1719
    364 merchant1514
    365 spammer1738
    366 buyer234
    367 merchant523
    368 buyer1926
    369 merchant1395
    370 merchant899
    371 merchant10
    372 buyer3
    373 buyer1639
    374 buyer1946
    375 merchant227
    376 merchant1409
    377 buyer1935
    378 merchant1801
    379 buyer556
    380 merchant1158
    381 merchant1512
    382 buyer1525
    383 merchant202
    384 buyer1088
    385 merchant1758
    386 buyer317
    387 buyer558
    388 spammer586
    389 merchant1612
    390 merchant94
    391 buyer1098
    392 merchant651
    393 buyer1431
    394 buyer1460
    395 merchant1766
    396 merchant1168
    397 merchant1142
    398 merchant745
    399 buyer1804
    400 buyer568
    401 buyer1202
    402 merchant880
    403 buyer991
    404 buyer632
    405 merchant211
    406 merchant1820
    407 buyer1285
    408 buyer569
    409 buyer196
    410 buyer1739
    411 buyer1996
    412 buyer46
    413 merchant594
    414 merchant1307
    415 merchant165
    416 merchant135
    417 merchant136
    418 buyer138
    419 buyer148
    420 merchant912
    421 buyer371
    422 merchant1181
    423 merchant800
    424 buyer1682
    425 buyer998
    426 merchant892
    427 merchant1444
    428 buyer392
    429 buyer1704
    430 buyer1413
    431 buyer671
    432 merchant1701
    433 merchant1648
    434 merchant1621
    435 merchant162
    436 merchant1845
    437 buyer905
    438 merchant1800
    439 merchant967
    440 buyer95
    441 merchant747
    442 buyer1891
    443 merchant658
    444 buyer459
    445 merchant616
    446 buyer703
    447 merchant174
    448 merchant1302
    449 buyer1148
    450 merchant1176
    451 merchant1871
    452 merchant1345
    453 buyer561
    454 buyer188
    455 merchant659
    456 merchant1517
    457 merchant992
    458 buyer1849
    459 merchant243
    460 buyer1209
    461 buyer1721
    462 merchant61
    463 merchant875
    464 merchant1502
    465 buyer1976
    466 merchant611
    467 spammer535
    468 merchant1513
    469 buyer1297
    470 buyer1864
    471 merchant1366
    472 merchant833
    473 merchant187
    474 merchant14
    475 merchant1488
    476 buyer1853
    477 buyer1817
    478 merchant69
    479 buyer440
    480 merchant235
    481 merchant1466
    482 merchant1779
    483 buyer1567
    484 merchant944
    485 merchant486
    486 buyer1878
    487 buyer640
    488 merchant664
    489 merchant741
    490 merchant27
    491 merchant414
    492 merchant839
    493 buyer1138
    494 buyer484
    495 merchant1790
    496 merchant1048
    497 merchant1272
    498 buyer803
    499 buyer1828
    500 merchant1361
    501 buyer214
    502 buyer1560
    503 buyer318
    504 merchant1743
    505 merchant247
    506 buyer1471
    507 buyer1783
    508 merchant1705
    509 merchant230
    510 merchant1887
    511 buyer1178
    512 buyer197
    513 buyer1075
    514 merchant828
    515 buyer303
    516 merchant310
    517 merchant980
    518 merchant614
    519 merchant1362
    520 merchant781
    521 buyer691
    522 merchant1370
    523 merchant1698
    524 buyer1747
    525 merchant585
    526 merchant248
    527 buyer1487
    528 buyer1078
    529 buyer1540
    530 merchant949
    531 merchant1750
    532 buyer1059
    533 merchant289
    534 buyer746
    535 buyer854
    536 buyer232
    537 merchant1874
    538 buyer271
    539 merchant858
    540 merchant773
    541 buyer1291
    542 merchant929
    543 merchant1964
    544 buyer553
    545 merchant1150
    546 buyer4
    547 buyer1620
    548 buyer925
    549 merchant1474
    550 merchant1208
    551 merchant1695
    552 merchant1360
    553 merchant511
    554 buyer1962
    555 buyer109
    556 buyer1264
    557 merchant250
    558 buyer994
    559 merchant96
    560 merchant567
    561 buyer775
    562 merchant865
    563 buyer1948
    564 merchant1840
    565 buyer1945
    566 merchant1749
    567 merchant1479
    568 merchant1823
    569 buyer1014
    570 merchant769
    571 buyer1152
    572 merchant960
    573 merchant1279
    574 merchant1394
    575 merchant1590
    576 buyer501
    577 merchant696
    578 buyer1896
    579 merchant1774
    580 buyer433
    581 buyer689
    582 buyer198
    583 merchant45
    584 buyer1799
    585 attacker1483
    586 merchant1528
    587 merchant1174
    588 merchant1784
    589 buyer1906
    590 merchant507
    591 merchant1811
    592 merchant1327
    593 merchant44
    594 merchant1110
    595 buyer1237
    596 merchant530
    597 merchant1741
    598 buyer1635
    599 buyer804
    600 merchant397
    601 buyer112
    602 buyer1835
    603 buyer1503
    604 buyer266
    605 buyer1971
    606 buyer885
    607 buyer788
    608 merchant228
    609 buyer1524
    610 merchant1765
    611 merchant387
    612 merchant1113
    613 merchant396
    614 merchant1249
    615 merchant576
    616 merchant1402
    617 merchant522
    618 buyer81
    619 merchant1623
    620 merchant1095
    621 buyer1894
    622 merchant1798
    623 buyer1649
    624 buyer915
    625 buyer1247
    626 buyer1175
    627 merchant454
    628 merchant1619
    629 merchant1978
    630 merchant1552
    631 merchant823
    632 merchant872
    633 buyer1434
    634 merchant1188
    635 merchant1038
    636 merchant1223
    637 merchant959
    638 merchant1764
    639 buyer1886
    640 merchant1338
    641 buyer969
    642 merchant226
    643 buyer1884
    644 buyer1955
    645 merchant1685
    646 buyer591
    647 buyer1848
    648 merchant1634
    649 merchant744
    650 buyer641
    651 buyer372
    652 merchant1146
    653 buyer1127
    654 merchant1928
    655 buyer620
    656 buyer1476
    657 buyer291
    658 merchant832
    659 merchant194
    660 merchant829
    661 merchant428
    662 buyer1097
    663 buyer1856
    664 merchant1717
    665 merchant1061
    666 buyer364
    667 buyer1586
    668 buyer1020
    669 merchant1511
    670 merchant1707
    671 buyer1397
    672 merchant1033
    673 merchant765
    674 merchant1541
    675 buyer1236
    676 buyer1872
    677 merchant352
    678 merchant698
    679 buyer1573
    680 buyer185
    681 merchant186
    682 merchant1693
    683 buyer1364
    684 buyer1486
    685 merchant1923
    686 buyer305
    687 buyer1461
    688 merchant1942
    689 buyer1380
    690 merchant71
    691 merchant1183
    692 buyer87
    693 merchant930
    694 merchant1794
    695 buyer9
    696 buyer268
    697 merchant275
    698 buyer6
    699 fraudster1286
    700 buyer1970
    701 merchant813
    702 merchant438
    703 buyer1320
    704 merchant789
    705 buyer1846
    706 merchant339
    707 merchant636
    708 buyer670
    709 buyer1677
    710 merchant285
    711 merchant322
    712 buyer1745
    713 merchant166
    714 merchant1287
    715 merchant377
    716 merchant862
    717 merchant1371
    718 merchant62
    719 buyer452
    720 buyer1136
    721 merchant657
    722 buyer1102
    723 buyer155
    724 buyer1617
    725 buyer1346
    726 buyer1636
    727 merchant1837
    728 buyer857
    729 buyer1388
    730 buyer690
    731 merchant739
    732 merchant1527
    733 merchant581
    734 buyer1508
    735 merchant855
    736 buyer544
    737 merchant1769
    738 merchant436
    739 merchant1106
    740 merchant450
    741 merchant993
    742 merchant601
    743 buyer1947
    744 buyer423
    745 buyer1559
    746 buyer529
    747 merchant1729
    748 buyer153
    749 buyer705
    750 buyer988
    751 buyer1596
    752 buyer1744
    753 buyer1898
    754 buyer1373
    755 merchant1164
    756 buyer683
    757 merchant1958
    758 merchant1530
    759 buyer1566
    760 merchant395
    761 merchant1177
    762 buyer48
    763 merchant920
    764 merchant679
    765 merchant240
    766 merchant1666
    767 buyer1355
    768 merchant164
    769 buyer838
    770 merchant1047
    771 buyer1777
    772 merchant1468
    773 buyer1056
    774 merchant1936
    775 buyer1583
    776 buyer267
    777 buyer1959
    778 merchant1574
    779 buyer134
    780 buyer1675
    781 buyer1847
    782 merchant1869
    783 buyer695
    784 merchant570
    785 merchant374
    786 merchant1727
    787 buyer506
    788 merchant758
    789 buyer499
    790 buyer47
    791 buyer897
    792 buyer487
    793 merchant320
    794 merchant1522
    795 buyer1980
    796 merchant996
    797 buyer1318
    798 merchant1881
    799 merchant1692
    800 merchant411
    801 buyer1231
    802 merchant667
    803 merchant1860
    804 buyer1266
    805 buyer1224
    806 buyer116
    807 buyer1333
    808 merchant245
    809 buyer1334
    810 buyer1753
    811 merchant492
    812 merchant646
    813 buyer1372
    814 buyer904
    815 buyer1432
    816 buyer1822
    817 buyer618
    818 merchant1206
    819 buyer1728
    820 merchant1356
    821 buyer861
    822 buyer602
    823 merchant33
    824 buyer1863
    825 merchant1053
    826 buyer1580
    827 buyer1983
    828 buyer160
    829 merchant104
    830 buyer1253
    831 buyer1184
    832 buyer1960
    833 buyer656
    834 merchant1123
    835 buyer1016
    836 buyer280
    837 merchant1029
    838 buyer237
    839 merchant470
    840 merchant856
    841 buyer108
    842 merchant146
    843 buyer1262
    844 buyer242
    845 merchant1007
    846 buyer441
    847 merchant375
    848 merchant1915
    849 merchant1903
    850 merchant1995
    851 merchant1818
    852 buyer1309
    853 merchant1278
    854 buyer68
    855 buyer421
    856 buyer928
    857 buyer149
    858 merchant264
    859 merchant1725
    860 merchant931
    861 buyer376
    862 merchant114
    863 buyer1068
    864 merchant212
    865 merchant782
    866 buyer78
    867 buyer1553
    868 merchant1714
    869 buyer25
    870 buyer255
    871 fraudster1083
    872 buyer1629
    873 merchant1643
    874 merchant124
    875 buyer1603
    876 merchant1653
    877 merchant251
    878 merchant678
    879 buyer515
    880 merchant547
    881 buyer1348
    882 buyer1057
    883 merchant1505
    884 merchant1330
    885 buyer1282
    886 buyer608
    887 buyer660
    888 merchant685
    889 buyer424
    890 merchant1688
    891 merchant223
    892 merchant1816
    893 merchant1651
    894 buyer1699
    895 merchant7
    896 merchant37
    897 merchant1600
    898 buyer1564
    899 merchant922
    900 merchant1077
    901 buyer1786
    902 buyer1027
    903 merchant1939
    904 buyer1137
    905 merchant837
    906 merchant626
    907 buyer302
    908 merchant1940
    909 buyer1267
    910 buyer1205
    911 buyer1482
    912 merchant950
    913 buyer101
    914 merchant539
    915 buyer787
    916 merchant1405
    917 buyer948
    918 buyer1257
    919 merchant1117
    920 merchant1422
    921 buyer1770
    922 buyer256
    923 merchant893
    924 merchant1423
    925 buyer709
    926 buyer1407
    927 buyer1426
    928 merchant1384
    929 buyer519
    930 merchant244
    931 buyer126
    932 merchant1041
    933 buyer456
    934 buyer1186
    935 buyer1913
    936 buyer923
    937 merchant1274
    938 buyer1997
    939 merchant461
    940 buyer1469
    941 buyer1193
    942 merchant287
    943 merchant976
    944 merchant1160
    945 merchant129
    946 buyer815
    947 buyer16
    948 merchant1532
    949 merchant1133
    950 buyer1159
    951 merchant278
    952 buyer1631
    953 buyer422
    954 buyer1203
    955 buyer1463
    956 buyer508
    957 merchant209
    958 buyer435
    959 merchant1167
    960 buyer315
    961 merchant1791
    962 merchant557
    963 attacker1496
    964 buyer1900
    965 merchant668
    966 buyer191
    967 merchant1658
    968 merchant1392
    969 fraudster1336
    970 buyer615
    971 merchant1182
    972 merchant536
    973 buyer808
    974 merchant496
    975 buyer827
    976 buyer1447
    977 merchant824
    978 buyer34
    979 merchant692
    980 merchant1080
    981 merchant1951
    982 merchant531
    983 buyer133
    984 merchant1051
    985 merchant220
    986 merchant399
    987 buyer171
    988 merchant1445
    989 buyer1711
    990 buyer1378
    991 merchant1834
    992 buyer1353
    993 buyer1396
    994 merchant1215
    995 buyer217
    996 buyer1221
    997 buyer914
    998 buyer1652
    999 merchant982
    1000 merchant1119
    1001 buyer942
    1002 merchant1458
    1003 buyer1009
    1004 buyer292
    1005 buyer458
    1006 buyer1974
    1007 buyer1385
    1008 merchant1589
    1009 merchant172
    1010 buyer439
    1011 buyer1124
    1012 buyer1428
    1013 merchant21
    1014 merchant406
    1015 merchant1998
    1016 merchant1944
    1017 merchant1039
    1018 merchant941
    1019 buyer1157
    1020 merchant702
    1021 merchant1390
    1022 merchant1225
    1023 buyer54
    1024 merchant1328
    1025 merchant891
    1026 buyer1633
    1027 buyer403
    1028 merchant1163
    1029 buyer1198
    1030 merchant1544
    1031 buyer120
    1032 buyer1433
    1033 buyer831
    1034 merchant337
    1035 merchant386
    1036 merchant1576
    1037 buyer1212
    1038 merchant1832
    1039 merchant190
    1040 merchant901
    1041 merchant848
    1042 buyer401
    1043 buyer1195
    1044 merchant1304
    1045 buyer1058
    1046 merchant617
    1047 merchant24
    1048 merchant792
    1049 buyer1311
    1050 merchant866
    1051 buyer332
    1052 buyer295
    1053 merchant1350
    1054 buyer85
    1055 merchant307
    1056 buyer552
    1057 merchant953
    1058 merchant1893
    1059 merchant1664
    1060 buyer1812
    1061 merchant574
    1062 buyer774
    1063 buyer379
    1064 buyer849
    1065 buyer1930
    1066 buyer1657
    1067 merchant654
    1068 bot1842
    1069 merchant39
    1070 merchant1671
    1071 buyer272
    1072 merchant143
    1073 buyer1464
    1074 buyer65
    1075 merchant1012
    1076 merchant888
    1077 merchant309
    1078 buyer1595
    1079 buyer475
    1080 merchant1477
    1081 merchant1954
    1082 buyer1630
    1083 merchant777
    1084 merchant1103
    1085 buyer483
    1086 merchant512
    1087 merchant1485
    1088 merchant1293
    1089 buyer1969
    1090 buyer1211
    1091 merchant1889
    1092 buyer1000
    1093 merchant1767
    1094 buyer1325
    1095 merchant605
    1096 buyer1343
    1097 merchant1037
    1098 merchant1156
    1099 merchant1819
    1100 buyer1276
    1101 merchant1235
    1102 buyer1625
    1103 merchant1430
    1104 buyer1462
    1105 buyer1449
    1106 buyer79
    1107 merchant1319
    1108 buyer688
    1109 buyer325
    1110 buyer22
    1111 buyer1096
    1112 merchant1204
    1113 merchant1418
    1114 buyer763
    1115 merchant84
    1116 buyer1134
    1117 merchant791
    1118 merchant592
    1119 merchant795
    1120 buyer152
    1121 merchant1808
    1122 merchant723
    1123 buyer1419
    1124 merchant882
    1125 merchant274
    1126 buyer1296
    1127 buyer1918
    1128 merchant879
    1129 merchant1994
    1130 merchant1805
    1131 merchant1003
    1132 buyer612
    1133 buyer168
    1134 merchant1399
    1135 merchant913
    1136 merchant413
    1137 merchant603
    1138 merchant1310
    1139 buyer1516
    1140 buyer771
    1141 merchant532
    1142 spammer55
    1143 buyer466
    1144 merchant1919
    1145 buyer588
    1146 buyer1875
    1147 merchant1315
    1148 merchant593
    1149 merchant469
    1150 buyer642
    1151 buyer1065
    1152 buyer961
    1153 merchant711
    1154 merchant1478
    1155 buyer1731
    1156 merchant239
    1157 merchant1965
    1158 merchant1909
    1159 buyer786
    1160 merchant1549
    1161 buyer238
    1162 buyer260
    1163 buyer573
    1164 buyer18
    1165 merchant367
    1166 merchant526
    1167 merchant5
    1168 buyer783
    1169 merchant383
    1170 merchant180
    1171 buyer835
    1172 buyer1515
    1173 buyer1490
    1174 merchant316
    1175 merchant1558
    1176 merchant1867
    1177 merchant1298
    1178 buyer40
    1179 merchant1001
    1180 merchant761
    1181 merchant360
    1182 buyer533
    1183 buyer541
    1184 buyer722
    1185 merchant1313
    1186 merchant497
    1187 buyer405
    1188 buyer345
    1189 buyer1904
    1190 buyer631
    1191 buyer1593
    1192 buyer204
    1193 buyer1736
    1194 merchant1562
    1195 merchant772
    1196 buyer676
    1197 buyer1473
    1198 merchant1902
    1199 buyer1480
    1200 merchant1070
    1201 buyer1321
    1202 buyer1387
    1203 merchant394
    1204 merchant926
    1205 buyer1092
    1206 merchant460
    1207 merchant1032
    1208 buyer141
    1209 buyer793
    1210 merchant927
    1211 merchant1703
    1212 merchant1344
    1213 merchant1210
    1214 merchant686
    1215 buyer1087
    1216 merchant1425
    1217 merchant1536
    1218 merchant1755
    1219 buyer1624
    1220 merchant465
    1221 merchant391
    1222 merchant1421
    1223 merchant478
    1224 merchant731
    1225 merchant937
    1226 buyer127
    1227 buyer1550
    1228 buyer600
    1229 merchant947
    1230 merchant1256
    1231 buyer179
    1232 merchant210
    1233 buyer128
    1234 merchant1993
    1235 buyer1554
    1236 merchant1147
    1237 buyer1792
    1238 merchant1877
    1239 merchant82
    1240 merchant246
    1241 merchant20
    1242 buyer1712
    1243 merchant177
    1244 buyer1140
    1245 buyer628
    1246 merchant1074
    1247 buyer1905
    1248 merchant1756
    1249 merchant0
    1250 buyer830
    1251 merchant738
    1252 buyer727
    1253 merchant965
    1254 merchant521
    1255 buyer1952
    1256 buyer1456
    1257 merchant1232
    1258 merchant161
    1259 buyer1091
    1260 merchant840
    1261 buyer1533
    1262 buyer736
    1263 buyer1637
    1264 buyer446
    1265 merchant814
    1266 merchant697
    1267 buyer1690
    1268 merchant661
    1269 buyer604
    1270 buyer1026
    1271 buyer740
    1272 merchant1599
    1273 merchant1888
    1274 buyer718
    1275 merchant1354
    1276 buyer674
    1277 buyer1854
    1278 buyer1851
    1279 buyer1034
    1280 buyer1581
    1281 merchant1700
    1282 buyer1977
    1283 merchant490
    1284 merchant1565
    1285 buyer1826
    1286 merchant1539
    1287 buyer794
    1288 buyer713
    1289 merchant900
    1290 buyer402
    1291 buyer1957
    1292 merchant1601
    1293 buyer388
    1294 merchant1323
    1295 buyer1572
    1296 buyer1892
    1297 buyer735
    1298 buyer1352
    1299 buyer649
    1300 buyer1497
    1301 merchant514
    1302 buyer276
    1303 merchant1689
    1304 merchant1571
    1305 buyer1696
    1306 merchant864
    1307 merchant546
    1308 merchant715
    1309 merchant1484
    1310 buyer400
    1311 buyer906
    1312 buyer729
    1313 buyer1268
    1314 merchant236
    1315 buyer277
    1316 buyer1060
    1317 merchant550
    1318 merchant582
    1319 merchant1882
    1320 merchant1365
    1321 merchant516
    1322 buyer1647
    1323 merchant625
    1324 merchant1455
    1325 buyer513
    1326 buyer717
    1327 merchant90
    1328 merchant1376
    1329 merchant1368
    1330 merchant841
    1331 buyer357
    1332 merchant1135
    1333 merchant629
    1334 merchant1270
    1335 buyer714
    1336 buyer107
    1337 buyer737
    1338 merchant1283
    1339 buyer51
    1340 merchant1417
    1341 buyer282
    1342 buyer1308
    1343 merchant1494
    1344 merchant970
    1345 merchant1213
    1346 merchant1694
    1347 merchant1988
    1348 merchant1785
    1349 buyer319
    1350 buyer1831
    1351 merchant1443
    1352 merchant1316
    1353 buyer140
    1354 buyer144
    1355 merchant634
    1356 buyer296
    1357 buyer606
    1358 buyer258
    1359 buyer1778
    1360 buyer1241
    1361 buyer1716
    1362 merchant817
    1363 merchant1406
    1364 buyer1073
    1365 buyer1655
    1366 buyer1379
    1367 buyer527
    1368 merchant1230
    1369 buyer390
    1370 buyer184
    1371 buyer338
    1372 merchant1857
    1373 merchant131
    1374 merchant1179
    1375 merchant252
    1376 buyer834
    1377 buyer750
    1378 buyer1416
    1379 buyer1859
    1380 merchant732
    1381 merchant163
    1382 buyer1715
    1383 merchant555
    1384 merchant58
    1385 merchant1932
    1386 buyer429
    1387 merchant1735
    1388 buyer64
    1389 buyer587
    1390 merchant916
    1391 merchant1999
    1392 buyer1899
    1393 merchant1979
    1394 merchant1943
    1395 buyer262
    1396 merchant1115
    1397 merchant1192
    1398 merchant1561
    1399 merchant1678
    1400 buyer935
    1401 buyer1082
    1402 buyer1827
    1403 buyer1720
    1404 buyer504
    1405 merchant797
    1406 buyer801
    1407 buyer1008
    1408 merchant1782
    1409 buyer1251
    1410 merchant1683
    1411 buyer1880
    1412 buyer1776
    1413 merchant1697
    1414 buyer1042
    1415 merchant1025
    1416 buyer1907
    1417 buyer989
    1418 merchant1615
    1419 merchant610
    1420 buyer1672
    1421 buyer845
    1422 buyer1108
    1423 merchant1284
    1424 buyer1457
    1425 buyer589
    1426 merchant825
    1427 buyer525
    1428 buyer1453
    1429 buyer1358
    1430 buyer407
    1431 merchant1084
    1432 buyer1751
    1433 buyer975
    1434 merchant1723
    1435 merchant1972
    1436 merchant56
    1437 merchant1269
    1438 merchant447
    1439 buyer1674
    1440 buyer699
    1441 buyer1132
    1442 buyer254
    1443 merchant1843
    1444 merchant877
    1445 merchant1641
    1446 merchant798
    1447 merchant997
    1448 buyer1844
    1449 buyer93
    1450 buyer1404
    1451 buyer811
    1452 merchant1890
    1453 merchant1131
    1454 merchant43
    1455 buyer1575
    1456 buyer873
    1457 merchant312
    1458 merchant42
    1459 merchant1363
    1460 merchant760
    1461 buyer1218
    1462 merchant1981
    1463 merchant1341
    1464 merchant1520
    1465 buyer249
    1466 merchant269
    1467 buyer776
    1468 merchant261
    1469 buyer1797
    1470 buyer1201
    1471 merchant538
    1472 buyer1626
    1473 buyer1506
    1474 buyer416
    1475 merchant1929
    1476 merchant1495
    1477 merchant361
    1478 buyer115
    1479 merchant1543
    1480 merchant63
    1481 buyer1667
    1482 merchant910
    1483 merchant1459
    1484 buyer1220
    1485 merchant909
    1486 merchant951
    1487 buyer455
    1488 buyer1815
    1489 merchant1870
    1490 merchant1243
    1491 merchant1732
    1492 merchant341
    1493 merchant734
    1494 merchant253
    1495 buyer122
    1496 merchant157
    1497 buyer1737
    1498 merchant417
    1499 merchant434
    1500 merchant924
    1501 buyer488
    1502 merchant1609
    1503 buyer1866
    1504 merchant125
    1505 buyer1375
    1506 merchant742
    1507 merchant1989
    1508 merchant1669
    1509 merchant1465
    1510 merchant933
    1511 buyer1519
    1512 merchant1109
    1513 buyer597
    1514 buyer753
    1515 merchant1153
    1516 merchant334
    1517 merchant259
    1518 buyer173
    1519 merchant1359
    1520 merchant1879
    1521 merchant297
    1522 merchant1602
    1523 merchant1023
    1524 merchant482
    1525 merchant1066
    1526 buyer1809
    1527 buyer820
    1528 buyer1374
    1529 buyer1795
    1530 buyer1099
    1531 merchant206
    1532 buyer1622
    1533 merchant964
    1534 buyer151
    1535 merchant946
    1536 merchant1548
    1537 buyer1531
    1538 buyer1100
    1539 buyer1197
    1540 merchant159
    1541 merchant707
    1542 buyer1594
    1543 buyer1557
    1544 buyer802
    1545 merchant1105
    1546 buyer183
    1547 buyer1933
    1548 buyer764
    1549 buyer1814
    1550 merchant528
    1551 buyer1015
    1552 merchant577
    1553 buyer1901
    1554 merchant757
    1555 buyer29
    1556 merchant1055
    1557 buyer382
    1558 merchant1401
    1559 buyer756
    1560 buyer118
    1561 buyer1437
    1562 buyer1233
    1563 merchant549
    1564 merchant15
    1565 merchant1841
    1566 merchant559
    1567 buyer1796
    1568 buyer749
    1569 buyer1337
    1570 merchant1126
    1571 buyer726
    1572 merchant1813
    1573 merchant1953
    1574 merchant1024
    1575 merchant537
    1576 buyer1312
    1577 merchant851
    1578 merchant1245
    1579 buyer911
    1580 merchant1013
    1581 merchant1761
    1582 buyer467
    1583 merchant1
    1584 buyer1154
    1585 merchant363
    1586 merchant896
    1587 buyer476
    1588 merchant807
    1589 merchant1668
    1590 buyer540
    1591 buyer1803
    1592 merchant110
    1593 buyer1555
    1594 merchant673
    1595 buyer1654
    1596 buyer1773
    1597 buyer1226
    1598 buyer1415
    1599 buyer370
    1600 buyer1044
    1601 merchant1398
    1602 merchant517
    1603 merchant748
    1604 buyer1259
    1605 buyer169
    1606 buyer1920
    1607 buyer437
    1608 merchant1781
    1609 buyer1111
    1610 buyer1149
    1611 buyer1523
    1612 merchant1467
    1613 buyer1638
    1614 buyer323
    1615 buyer1254
    1616 merchant427
    1617 buyer1481
    1618 buyer1504
    1619 merchant1606
    1620 merchant366
    1621 buyer453
    1622 merchant1389
    1623 bot340
    1624 merchant150
    1625 buyer919
    1626 buyer205
    1627 merchant1757
    1628 buyer1660
    1629 buyer895
    1630 merchant903
    1631 buyer1938
    1632 buyer193
    1633 buyer378
    1634 merchant1403
    1635 buyer852
    1636 merchant444
    1637 merchant1937
    1638 merchant286
    1639 merchant590
    1640 merchant410
    1641 buyer1118
    1642 buyer1240
    1643 merchant1563
    1644 merchant1207
    1645 merchant281
    1646 merchant1442
    1647 buyer1335
    1648 merchant842
    1649 merchant336
    1650 bot1967
    1651 merchant60
    1652 buyer1850
    1653 merchant1510
    1654 merchant218
    1655 buyer2
    1656 buyer1441
    1657 buyer719
    1658 merchant77
    1659 buyer199
    1660 merchant1288
    1661 merchant1010
    1662 buyer493
    1663 buyer121
    1664 buyer1261
    1665 merchant785
    1666 buyer130
    1667 merchant1568
    1668 buyer1424
    1669 merchant1493
    1670 buyer384
    1671 merchant474
    1672 merchant463
    1673 merchant1222
    1674 buyer607
    1675 buyer1196
    1676 buyer479
    1677 merchant863
    1678 merchant985
    1679 buyer1914
    1680 merchant356
    1681 merchant330
    1682 merchant1093
    1683 merchant580
    1684 buyer805
    1685 buyer1086
    1686 merchant409
    1687 buyer1742
    1688 merchant1730
    1689 merchant1349
    1690 buyer1537
    1691 buyer622
    1692 buyer1663
    1693 buyer76
    1694 merchant1645
    1695 buyer331
    1696 merchant347
    1697 buyer358
    1698 buyer1200
    1699 merchant448
    1700 buyer1861
    1701 buyer878
    1702 merchant156
    1703 buyer638
    1704 buyer1836
    1705 buyer973
    1706 merchant874
    1707 merchant1435
    1708 merchant233
    1709 merchant1295
    1710 merchant1640
    1711 merchant1386
    1712 merchant1924
    1713 buyer80
    1714 merchant759
    1715 merchant1975
    1716 merchant1229
    1717 merchant1876
    1718 merchant1646
    1719 merchant1825
    1720 buyer1173
    1721 merchant306
    1722 merchant1451
    1723 merchant869
    1724 buyer999
    1725 merchant1911
    1726 buyer425
    1727 merchant1369
    1728 buyer457
    1729 buyer11
    1730 merchant398
    1731 merchant1501
    1732 merchant1680
    1733 fraudster936
    1734 buyer1491
    1735 merchant1613
    1736 merchant1627
    1737 merchant1072
    1738 buyer1771
    1739 buyer1570
    1740 buyer725
    1741 merchant299
    1742 merchant1125
    1743 merchant1289
    1744 buyer716
    1745 merchant1821
    1746 buyer917
    1747 buyer974
    1748 buyer963
    1749 buyer181
    1750 merchant147
    1751 buyer543
    1752 merchant1873
    1753 merchant1921
    1754 buyer257
    1755 merchant355
    1756 buyer1069
    1757 merchant1470
    1758 buyer1189
    1759 buyer1021
    1760 buyer1290
    1761 buyer1114
    1762 buyer1217
    1763 buyer73
    1764 merchant1252
    1765 buyer943
    1766 merchant1227
    1767 buyer687
    1768 merchant1357
    1769 buyer1810
    1770 merchant1275
    1771 buyer710
    1772 merchant884
    1773 merchant1865
    1774 buyer1961
    1775 buyer265
    1776 buyer1101
    1777 attacker1448
    1778 buyer1317
    1779 merchant1166
    1780 merchant241
    1781 merchant983
    1782 merchant1535
    1783 buyer1852
    1784 merchant412
    1785 merchant1956
    1786 merchant1299
    1787 merchant31
    1788 merchant1927
    1789 buyer613
    1790 merchant627
    1791 merchant59
    1792 merchant819
    1793 attacker362
    1794 merchant19
    1795 merchant1022
    1796 buyer1591
    1797 merchant1908
    1798 buyer300
    1799 buyer940
    1800 merchant1973
    1801 buyer1661
    1802 buyer1984
    1803 merchant1377
    1804 buyer1551
    1805 merchant995
    1806 merchant219
    1807 buyer966
    1808 merchant1420
    1809 merchant545
    1810 merchant1910
    1811 merchant921
    1812 buyer806
    1813 merchant1604
    1814 merchant1214
    1815 buyer1748
    1816 merchant167
    1817 buyer981
    1818 buyer1180
    1819 merchant1234
    1820 merchant418
    1821 merchant1584
    1822 buyer1673
    1823 merchant1239
    1824 buyer799
    1825 merchant426
    1826 merchant1242
    1827 buyer119
    1828 buyer1547
    1829 buyer1248
    1830 merchant826
    1831 buyer1439
    1832 buyer652
    1833 merchant74
    1834 buyer351
    1835 merchant1925
    1836 buyer1632
    1837 buyer451
    1838 merchant644
    1839 merchant1155
    1840 merchant1002
    1841 buyer883
    1842 merchant1144
    1843 buyer88
    1844 buyer770
    1845 merchant195
    1846 merchant945
    1847 merchant1141
    1848 buyer876
    1849 merchant354
    1850 buyer449
    1851 merchant1950
    1852 merchant1094
    1853 buyer562
    1854 merchant1049
    1855 merchant1475
    1856 merchant1656
    1857 buyer1931
    1858 merchant1006
    1859 buyer907
    1860 merchant1381
    1861 buyer720
    1862 merchant1760
    1863 merchant1588
    1864 buyer1081
    1865 merchant431
    1866 buyer701
    1867 merchant66
    1868 buyer986
    1869 buyer464
    1870 buyer26
    1871 buyer721
    1872 buyer1982
    1873 merchant381
    1874 buyer1400
    1875 merchant1710
    1876 buyer215
    1877 merchant1492
    1878 merchant784
    1879 buyer99
    1880 buyer693
    1881 buyer1139
    1882 merchant680
    1883 buyer554
    1884 merchant860
    1885 merchant222
    1886 merchant1294
    1887 buyer1228
    1888 buyer1793
    1889 merchant755
    1890 buyer1789
    1891 buyer662
    1892 merchant1277
    1893 merchant92
    1894 buyer578
    1895 merchant1858
    1896 merchant1120
    1897 buyer36
    1898 buyer666
    1899 merchant111
    1900 merchant821
    1901 buyer1301
    1902 buyer1411
    1903 buyer505
    1904 merchant1185
    1905 buyer1104
    1906 merchant1702
    1907 buyer1273
    1908 merchant1577
    1909 merchant571
    1910 buyer57
    1911 merchant1280
    1912 merchant353
    1913 merchant1130
    1914 merchant213
    1915 buyer881
    1916 buyer182
    1917 buyer809
    1918 buyer733
    1919 buyer724
    1920 buyer1862
    1921 buyer524
    1922 merchant682
    1923 merchant408
    1924 buyer17
    1925 merchant1367
    1926 merchant462
    1927 buyer481
    1928 buyer767
    1929 merchant1162
    1930 merchant480
    1931 buyer178
    1932 merchant1687
    1933 merchant1018
    1934 buyer850
    1935 merchant294
    1936 merchant1578
    1937 buyer751
    1938 merchant1324
    1939 buyer1412
    1940 merchant1244
    1941 merchant290
    1942 buyer818
    1943 merchant1592
    1944 buyer208
    1945 buyer348
    1946 merchant957
    1947 merchant1992
    1948 buyer868
    1949 merchant498
    1950 buyer270
    1951 merchant1043
    1952 buyer822
    1953 buyer846
    1954 buyer97
    1955 merchant1763
    1956 buyer503
    1957 buyer1161
    1958 buyer1538
    1959 merchant653
    1960 buyer1329
    1961 buyer977
    1962 merchant1452
    1963 buyer102
    1964 buyer489
    1965 merchant1916
    1966 merchant595
    1967 merchant75
    1968 merchant369
    1969 buyer1382
    1970 buyer655
    1971 merchant1436
    1972 buyer954
    1973 buyer1507
    1974 buyer100
    1975 buyer938
    1976 merchant955
    1977 buyer635
    1978 buyer142
    1979 buyer1618
    1980 buyer648
    1981 merchant1112
    1982 merchant1438
    1983 merchant304
    1984 buyer1754
    1985 merchant650
    1986 buyer221
    1987 buyer1706
    1988 merchant53
    1989 buyer1587
    1990 buyer333
    1991 buyer1498
    1992 merchant1255
    1993 buyer887
    1994 buyer1410
    1995 merchant137
    1996 buyer419
    1997 merchant624
    1998 buyer890
    1999 merchant293


## 5.3 Model Querying Functions
This next cell introduces a modified version of the classify_candidates_average function previously used. The main difference is that it will perform the classification with a much wider Kullback-Leibler divergence limit, and return a list of candidate instead of the most probable one, as well as the profile of the class for comparison. 


```python
def classify_candidate_query(candidate_name, behaviour_type_average, score, limit = 7):
    potential_matching_type = {}
    
    for be, be_average in behaviour_type_average.items():
        result = compare_profiles(score[candidate_name], be_average, limit)
            
        if result['test'] == True:
            potential_matching_type[be] = { 
                'group': be,
                'kldp': result['kldp'], 
                'kldq': result['kldq'], 
                'proximity': result['kldp']**2 + result['kldq']**2,
                'profile': behaviour_type_average[be]
            }

    if len(potential_matching_type.keys()) == 0:
        return {}
    else:
        matching_scores = {x: potential_matching_type[x]['proximity'] for x in potential_matching_type.keys()}
        best = min(matching_scores, key=matching_scores.get)
        return {
            'best': potential_matching_type[best],
            'others': potential_matching_type
        }

candidate = listing_of_users[699]
result = classify_candidate_query(candidate, behaviour_average_table, user_transition_score_merged, 20)
```


```python
# This is an ugly function that takes the candidate profile and the potential classes and present them in a report
# The first row of the report is the candidate (groupe -1), then we can compare each lines with the candidate line
# and see where the potential class differs from the candidate

def classify_candidate_detail(candidate, result, score):
    reports = []
    prefix = ['group', 'kldp', 'kdlq', 'proximity']
    headers = []
    column = {}
    
    for key, value in result['best']['profile'].items():
        if key not in prefix:
            column[key] = 1

    for key, value in score[candidate].items():
        if key not in prefix:
            column[key] = 1

    columns = sorted(column.keys())
    headers.extend(prefix)
    headers.extend(columns)

    ## Adding the candidate line in the report
    candidate_rows = [-1 for x in prefix]
    for action in columns: 
        candidate_rows.append(score[candidate][action])
    reports.append(candidate_rows)

    ## Adding all potential classes in the report
    for key, value in result['others'].items():
        rows = [key, value['kldp'], value['kldq'], value['proximity']]
        for action in columns: 
            rows.append(value['profile'][action])
        reports.append(rows)

    # transforming the array in a dataframe to simplify sorting
    df_report = pd.DataFrame(reports, columns=headers)

    return df_report.sort_values(by=['proximity'])

# Display the report
print('report for {}. score of {}'.format(candidate,sum(user_transition_score_merged[candidate].values())))
print('matching with behaviour group:', classify_candidates_average(candidate, behaviour_average_table, user_transition_score_merged, 20))

classify_candidate_detail(candidate, result, user_transition_score_merged)
```

    report for fraudster1286. score of 54.17083457486127
    matching with behaviour group: 9





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>kldp</th>
      <th>kdlq</th>
      <th>proximity</th>
      <th>bank_modify</th>
      <th>buy_item</th>
      <th>comment</th>
      <th>end</th>
      <th>home</th>
      <th>login</th>
      <th>logout</th>
      <th>password_reset</th>
      <th>payment_modify</th>
      <th>sell_item</th>
      <th>update_address</th>
      <th>update_email</th>
      <th>view_item</th>
      <th>view_profile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.0</td>
      <td>19.905455</td>
      <td>0.0</td>
      <td>10.959278</td>
      <td>0.0</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.584963</td>
      <td>0.000000</td>
      <td>5.357552</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>7.508719</td>
      <td>1.988827</td>
      <td>60.336293</td>
      <td>0.0</td>
      <td>12.238834</td>
      <td>0.0</td>
      <td>10.959278</td>
      <td>0.0</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.550333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>18.219256</td>
      <td>-0.569793</td>
      <td>332.265944</td>
      <td>0.0</td>
      <td>13.370899</td>
      <td>0.0</td>
      <td>10.959278</td>
      <td>0.0</td>
      <td>5.087463</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Note about the classification:
* When we build our behaviour group table, we use a smaller Kellback Leibler Divergence limit (5), but here, when querying, and to compare with other classes, we can affort to search a bit more wider. If used alone, this could results in false positives, so external factors should also be taken in consideration.

Note about how to read the Kellback Leibler Divergence P and Q columns (kdlp, kdlq):
* the smaller the number, the more the candidate ressemble to the target class, but this is not symetric, so the candidate can have similarities with the group, but be missing information as well. In that case, kldp will be bigger.
* The scores in the actions columns (bank_modify, buy_items, etc.) is the sum of all surprisal values for the actions taken by the users. A larger value means that the user did something unusual, or a large amount of usual actions.


```python
chrono_stop = timeit.default_timer()

print('Complete run time: ', chrono_stop - chrono_start) 

```

    Complete run time:  122.47667773801368

