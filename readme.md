
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
    CPU times: user 7.66 s, sys: 73.3 ms, total: 7.73 s
    Wall time: 7.94 s


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

    CPU times: user 6.28 s, sys: 41 ms, total: 6.32 s
    Wall time: 6.53 s


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

    CPU times: user 11.3 s, sys: 111 ms, total: 11.4 s
    Wall time: 11.9 s



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
```

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

candidate = listing_of_users[0]
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

    report for buyer495. score of 231.00757652010145
    matching with behaviour group: 3





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
      <td>43.143746</td>
      <td>8.303781</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>153.237185</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.313816</td>
      <td>0.428215</td>
      <td>0.281848</td>
      <td>0.0</td>
      <td>40.293834</td>
      <td>10.850554</td>
      <td>10.959278</td>
      <td>0.390689</td>
      <td>6.098043</td>
      <td>9.957565</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.648287</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.970284</td>
      <td>-0.019148</td>
      <td>3.882386</td>
      <td>0.0</td>
      <td>20.162354</td>
      <td>1.760402</td>
      <td>10.959278</td>
      <td>0.000047</td>
      <td>5.116970</td>
      <td>6.456489</td>
      <td>0.003725</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.159787</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3.794519</td>
      <td>0.060074</td>
      <td>14.401985</td>
      <td>0.0</td>
      <td>37.365853</td>
      <td>12.661244</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087519</td>
      <td>1.233135</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>118.797446</td>
      <td>0.053576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.992436</td>
      <td>0.395050</td>
      <td>64.035099</td>
      <td>0.0</td>
      <td>23.272177</td>
      <td>0.000000</td>
      <td>10.959278</td>
      <td>0.390787</td>
      <td>6.154024</td>
      <td>9.248512</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>59.854262</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>38</td>
      <td>10.263511</td>
      <td>0.017950</td>
      <td>105.339981</td>
      <td>0.0</td>
      <td>27.133896</td>
      <td>11.220432</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.665715</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>96.699037</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>32</td>
      <td>9.741091</td>
      <td>8.322777</td>
      <td>164.157477</td>
      <td>0.0</td>
      <td>25.886248</td>
      <td>8.303781</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>10.736805</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.807355</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>104.376825</td>
      <td>5.357552</td>
    </tr>
    <tr>
      <th>7</th>
      <td>30</td>
      <td>-0.437201</td>
      <td>13.300067</td>
      <td>177.082914</td>
      <td>0.0</td>
      <td>43.143746</td>
      <td>8.303781</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.584963</td>
      <td>90.630261</td>
      <td>10.715104</td>
    </tr>
    <tr>
      <th>11</th>
      <td>42</td>
      <td>-0.077888</td>
      <td>17.222899</td>
      <td>296.634304</td>
      <td>0.0</td>
      <td>25.886248</td>
      <td>8.303781</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.000000</td>
      <td>6.507795</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>104.376825</td>
      <td>10.715104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16</td>
      <td>6.989511</td>
      <td>16.997562</td>
      <td>337.770392</td>
      <td>0.0</td>
      <td>41.632800</td>
      <td>0.000000</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087463</td>
      <td>10.276124</td>
      <td>0.000000</td>
      <td>3.030626</td>
      <td>0.0</td>
      <td>1.292481</td>
      <td>0.646241</td>
      <td>59.785118</td>
      <td>12.054492</td>
    </tr>
    <tr>
      <th>9</th>
      <td>37</td>
      <td>18.574749</td>
      <td>-0.028823</td>
      <td>345.022143</td>
      <td>0.0</td>
      <td>15.902968</td>
      <td>0.000000</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.169741</td>
      <td>0.000000</td>
      <td>0.003322</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35.465197</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>23</td>
      <td>18.048348</td>
      <td>8.130342</td>
      <td>391.845313</td>
      <td>0.0</td>
      <td>17.257499</td>
      <td>0.000000</td>
      <td>10.959278</td>
      <td>0.000000</td>
      <td>5.087463</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.807355</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.805335</td>
      <td>5.357552</td>
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

    Complete run time:  115.2286783939926

