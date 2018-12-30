
# How to find and group outliers in a logstream, based on behaviour

## Context

Your boss asked you to find suspect users among the few thousands of customers using your web site every day.

Where do you start?! Easy! AI, Tensorflow, machine learning, and other buzzwords like so will solve the issue! When all logs are categorized, it's quite easy right?! You know what is going on in your logs right?

... most probably not, especially if your site is generating few gigabytes of logs per day, there is just too much logs to know everything that is going one.

Before even starting to search for suspicious users, step 1 would be starting by identifying what it means to be suspicious.

    Suspicious normally means that it stands out from the normal.
    
Ok, step 2: what is normal?

    Normal behaviour consist of expected usage. Anything outside of this can be considered abnormal.

... We are not much more advanced because now we need to go over each action, and identify what is normal, and what is not, and assign a score to each.

The second issue with this is that normal behaviour will be definition be way more likely than the abnormal one, which will skew the results into making any machine learning model think that 99.9% of the usage is normal, which a margin of error of + or - 5%, flooding your analysis with noise.

The classical approach is to use a simple rule based system. By setting few triggers, we can find out account attackers. For small scale or simple systems, this will is likely to be enough, but not for large scale and complex systems.

One other hypothesis would be to use Hidden Markov Chains, but for HNN, we need to know what we are searching for.
Logs are also noisy: users will not simply go from A to B then C. they might crawl the whole site before reaching C, which complicates the creation of HMM.

Another hypothesis would be to filter out known noise, like when we want to do sentiment analysis, English stop words are just too common and does not tell us anything useful... but what if these stop words are useless in small quantities, but an indicator of an issue in large quantities? ... and how much is too much?


## Information Theory To The Rescue

Information Theory is normally used for signal analysis and compression. By using Surprisal Analysis and Kullback–Leibler divergence, we can group similar behaviour togethers.

* [Information Content](https://en.wikipedia.org/wiki/Information_content)
* [Surprisal Analysis](https://en.wikipedia.org/wiki/Surprisal_analysis)
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
* [Example of Surprisal](http://www.umsl.edu/~fraundorfp/egsurpri.html)

Basically, Surprisal Analysis is a way to measure (in bits) how surprised we will be by looking at something. Surprisal are calculated using Log base 2, which makes it easy to calculate because log are additive when probabilities are multiplied.

For example:
* There is 1/6 chance to roll a 6 die, gives us 2.58 bits of information (-log2(1/6) ~= 2.4849 bits). If you are playing D&D, you will be happily surprised.
* Not rolling a 6 on a die represents 0.263 bits (-log2(5/6) ~= 0.263 bits). So rolling anything but a 6 in that same D&D game will most likely leave you dissapointed. 
* The chance of rolling 10 x 6 in a row on a die is 1/60466176 (10 * -log2(1/6) ~= 25.85 bits). This is highly unlikely.

By assigning a surprisal value to each action taken over all actions taken by all users gives us an idea how much surprisal by seeing any of them happening, and by adding the score of all actions of a user, we can get an idea how "normal" was all his/her actions.

This is what this notebook intend to demonstrate: Using Surprisal analysis, we will assign a score to actions, and by adding up the score of each action, identify series of actions that are unlikely to occur.

# But first, we need some logs

What is explained in this notebook can be applied to real logs, but for the experimentation, I added a "blackbox" library that will help to generate our users and the logs of these users based on a determined probability distribution.

## Caviat: I had to cheat to achieve the desired results

Because I am generating the logs through a known distribution, I am also able to know the type of user of each user, which makes it easy later on to debug and know how wrong I am.

This cheat was also necessary to get a probability distribution of the classification function of this notebook. This probability distribution is likely to be different for live environment, and should be tweeked.

| | Positive | Negative |
| -- | -- | -- |
| True | 0.875 | 0.102 |
| False | 0.125 | 0.898 |

I am using this probability distribution, along with the Bayes Theorem to update my belief that each tested candidates truely belong in the tested behaviour classification.

## Different User Profiles Generated By The Library

The following profiles are generated by the library.

Normal users
* Buyer
* Merchants

Abnormal users:
* Scraper bots
* Spammers
* fraudster
* Account Attackers

Buyers and merchants represent 98% of our logs. Leaving 2% to the abnormal users. However, the actions taken by each users being on a probability distribution, it is possible to see an "attakcer" user being classified as a user, because that attacked might have had a change of heart and didn't attack after all.


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

from blackbox import distribution
from blackbox import generate_userlist, generate_logs, cheat_calculate_hit_rate, cheat_lookup_all_users

magic = distribution()
```

# Generating The User Database

The following cell generates our user database.

You will notice in this notebook that we are reinitializing the random seed quite often, just to keep consistence during the testing. Set the *random_seed* to False to randomize everything.


```python
random_seed = 42

if random_seed:
    random.seed(random_seed)

## We define how many users here
number_of_new_users = 1000


existing_users = [] # Later on, we can add users to our list by supplying it to the generate_userlist function
user_lists = generate_userlist(number_of_new_users, existing_users)

print(len(user_lists), user_lists[:5])
```

    1000 ['merchant', 'buyer', 'buyer', 'buyer', 'merchant']


# Generating Logs For Day 1

Note: The more users we have, the more log events will be generated. The probability distribution of each user ensures that they will start with a defined action, crawl the site following a defined pattern, and logout eventually, until the end of the day.


```python
%%time
if random_seed:
    random.seed(random_seed)

start_time = datetime(2019,1,1,0,0)
day1_logs = generate_logs(user_lists, start_time)

print(len(day1_logs), 'logs event generated for', len(user_lists), 'users')
```

    19576 logs event generated for 1000 users
    CPU times: user 2.08 s, sys: 16.3 ms, total: 2.1 s
    Wall time: 2.11 s


## Transforming the logs in a pandas dataframe

The transition surprisal lookup table used in this notebook calculates scores based on the movements of the users between each actions. For example:

* login (success) -> view_items (success) will result in a low surpisal value
* login (fail) -> buy_item (success) never happened. If this sequence happen, this should be a huge red flag.


```python
def transform_logs_to_pandas(logs):
    data = pd.DataFrame(np.array(logs), columns=['time', 'user', 'path', 'status', 'uidx', 'realtype'])
    
    data['prev_path'] = data.groupby(['user'])['path'].shift(1)
    data['prev_path'] = data['prev_path'].fillna("")

    data['prev_status'] = data.groupby(['user'])['status'].shift(1)
    data['prev_status'] = data['prev_status'].fillna("")
    return data
    
day1_data = transform_logs_to_pandas(day1_logs)

print(day1_data.loc[(day1_data['path'] == 'login') & (day1_data['status'] == 'fail')].head())
```

                         time         user   path status uidx  realtype prev_path  \
    39    2019-01-01 00:05:50  merchant898  login   fail  898  merchant             
    1091  2019-01-01 02:04:15     buyer306  login   fail  306     buyer             
    1384  2019-01-01 02:30:16     buyer353  login   fail  353     buyer             
    1388  2019-01-01 02:30:38  attacker725  login   fail  725  attacker             
    1389  2019-01-01 02:30:39  attacker725  login   fail  725  attacker     login   
    
         prev_status  
    39                
    1091              
    1384              
    1388              
    1389        fail  


The following cell generates the transition surprisal lookup table used to score each actions taken by the users.

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
                'ssurprisal': log(1/(dsuccess / denum),2), 
                'fsurprisal': log(1/(dfail / denum),2),
            }
    return surprisal

transition_surprisal = init_transition_surprisal_lookup(day1_data, 'path', 'prev_path', 'status', 'success')
```


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

get_transition_surprisal('buy_item', 'login', transition_surprisal, day1_data)
```




    {'fail': 0, 'fsurprisal': 10.321928094887362, 'ssurprisal': 7.0, 'success': 10}




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

print(user_transition_score['fraudster96'])
print(user_transition_score['buyer402'])
```

    {'login': 5.162147650850593, 'view_item': 33.24141988768278, 'sell_item': 0, 'buy_item': 49.60964047443681, 'end': 9.962896005337262, 'logout': 0, 'view_profile': 4.857980995127573, 'update_address': 5.169925001442313, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 0, 'update_email': 0}
    {'login': 5.162147650850593, 'view_item': 25.5340607556019, 'sell_item': 0, 'buy_item': 9.321928094887362, 'end': 9.962896005337262, 'logout': 9.271463027904375, 'view_profile': 0, 'update_address': 0, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 0, 'update_email': 0}


If we go through the logs of the day, can we identify outliers?

To do so, the idea is to look at each action being taken by each user, and add the relevant value from the surprisal lookup table.

That said, I did read a lot about information theory and surprisal analysis, and this most probably not how it is supposed to be used, and the calculation is most probably wrong... but this mistake is quite useful


```python
cumulative_score = [[v,sum(user_transition_score[v].values())] for v in [k for k in list(user_transition_score.keys())]]

df_cumulative_score = pd.DataFrame(cumulative_score, columns=['user', 'surprisal'])

avg = df_cumulative_score['surprisal'].mean()
std = df_cumulative_score['surprisal'].std()
df_cumulative_score['z'] = (df_cumulative_score['surprisal'] - avg) / std

```


```python
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
      <th>721</th>
      <td>bot672</td>
      <td>32110.895325</td>
      <td>30.442508</td>
    </tr>
    <tr>
      <th>684</th>
      <td>bot900</td>
      <td>8122.097174</td>
      <td>7.589254</td>
    </tr>
    <tr>
      <th>275</th>
      <td>bot776</td>
      <td>2922.642534</td>
      <td>2.635923</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cumulative_score.sort_values(by=['surprisal'], ascending=False).tail(15)
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
      <th>800</th>
      <td>merchant722</td>
      <td>37.714803</td>
      <td>-0.112442</td>
    </tr>
    <tr>
      <th>761</th>
      <td>merchant102</td>
      <td>37.714803</td>
      <td>-0.112442</td>
    </tr>
    <tr>
      <th>756</th>
      <td>merchant121</td>
      <td>37.714803</td>
      <td>-0.112442</td>
    </tr>
    <tr>
      <th>749</th>
      <td>merchant780</td>
      <td>37.714803</td>
      <td>-0.112442</td>
    </tr>
    <tr>
      <th>157</th>
      <td>merchant973</td>
      <td>35.658602</td>
      <td>-0.114401</td>
    </tr>
    <tr>
      <th>102</th>
      <td>merchant221</td>
      <td>35.658602</td>
      <td>-0.114401</td>
    </tr>
    <tr>
      <th>924</th>
      <td>merchant376</td>
      <td>35.658602</td>
      <td>-0.114401</td>
    </tr>
    <tr>
      <th>52</th>
      <td>buyer416</td>
      <td>34.718435</td>
      <td>-0.115297</td>
    </tr>
    <tr>
      <th>310</th>
      <td>buyer608</td>
      <td>34.718435</td>
      <td>-0.115297</td>
    </tr>
    <tr>
      <th>303</th>
      <td>buyer663</td>
      <td>34.718435</td>
      <td>-0.115297</td>
    </tr>
    <tr>
      <th>996</th>
      <td>buyer324</td>
      <td>34.718435</td>
      <td>-0.115297</td>
    </tr>
    <tr>
      <th>998</th>
      <td>buyer528</td>
      <td>24.753975</td>
      <td>-0.124789</td>
    </tr>
    <tr>
      <th>807</th>
      <td>fraudster255</td>
      <td>24.396507</td>
      <td>-0.125130</td>
    </tr>
    <tr>
      <th>97</th>
      <td>attacker725</td>
      <td>18.993460</td>
      <td>-0.130277</td>
    </tr>
    <tr>
      <th>999</th>
      <td>merchant238</td>
      <td>5.162148</td>
      <td>-0.143454</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.seterr(divide='ignore', invalid='ignore', over='ignore')

if random_seed:
    np.random.seed(random_seed)

for l in range(2,8):
    flat_status, flat_lookup = cheat_calculate_hit_rate(day1_data, user_transition_score, l)
    print('limit',l)
    print('count', True, flat_status[True])
    print('count', False, flat_status[False])

    print('percent', True, flat_lookup[True])
    print('percent', False, flat_lookup[False])

flat_status, flat_lookup = cheat_calculate_hit_rate(day1_data, user_transition_score, 6)
```

    limit 2
    count True {True: 112, False: 0}
    count False {True: 147, False: 966}
    percent True {True: 1.0, False: 0.1320754716981132}
    percent False {True: 0.0, False: 0.8679245283018868}
    limit 3
    count True {True: 113, False: 0}
    count False {True: 146, False: 966}
    percent True {True: 1.0, False: 0.13129496402877697}
    percent False {True: 0.0, False: 0.8687050359712231}
    limit 4
    count True {True: 109, False: 0}
    count False {True: 150, False: 966}
    percent True {True: 1.0, False: 0.13440860215053763}
    percent False {True: 0.0, False: 0.8655913978494624}
    limit 5
    count True {True: 110, False: 2}
    count False {True: 149, False: 964}
    percent True {True: 0.9821428571428571, False: 0.13387241689128482}
    percent False {True: 0.017857142857142856, False: 0.8661275831087152}
    limit 6
    count True {True: 135, False: 4}
    count False {True: 124, False: 962}
    percent True {True: 0.9712230215827338, False: 0.1141804788213628}
    percent False {True: 0.02877697841726619, False: 0.8858195211786372}
    limit 7
    count True {True: 119, False: 4}
    count False {True: 140, False: 962}
    percent True {True: 0.967479674796748, False: 0.12704174228675136}
    percent False {True: 0.032520325203252036, False: 0.8729582577132486}


Can we cluster users together?


We will use two lists: 
- unclassified_users, which is a copy of the original user_score list, but not classified yet
- behaviour_types: this is a list of hashes, the key being the user type (normally type + n), and the value of that hash is an array with all the profiles of the users classified under that type

With these 2 lists, here is the idea:
- From the unclassified_users list, we will pick one at random
- Then we will pick 100 users at random, and try to find 10 that matches the behaviour. It's ok if we don't find any
- I make the hypothesis that we have 10 kind of users, so my prior probability is 0.1
- Then we check in all the keys under behaviour_types and compare our found random candidates with max 10 random candidates from that behaviour type. For the experimental data, We know that the True Positive probability is 51.5%, and the True Negative rate is 11.2%. By doing enough tests, we should be able to update our belief that we are comparing matching or different profiles


```python
def update_probability(prior_probability, distribution, test_result):
    
    # What is our success rate for this test_result?
    likelihood_of_being_right = distribution[test_result][True]
    likelihood_of_being_wrong = distribution[test_result][False]
          
    numerator = likelihood_of_being_right * prior_probability
    denominator = (likelihood_of_being_right * prior_probability) + (likelihood_of_being_wrong * (1 - prior_probability))
    
    posterior_probability = numerator / denominator
    
    return posterior_probability
```


```python
np.seterr(divide='ignore', invalid='ignore', over='ignore')

def compare_profiles(profile1, profile2, limit = 7):
    u1 = np.array(list(profile1.values()))
    u2 = np.array(list(profile2.values()))
    
    # Ref: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition
    px = [1/np.power(2,x) for x in u1]    
    qx = [1/np.power(2,x) for x in u2]
    
    p = np.array(qx)/np.array(px)
    q = np.array(px)/np.array(qx)
    dklp = (qx * np.log2(p)).sum()
    dklq = (px * np.log2(q)).sum()
    
    t = dklp < limit and dklp >= -limit and dklq < limit and dklq >= -limit
    
    return {'test': t, 'dklp': dklp, 'dklq': dklq}

print(user_transition_score['fraudster96'])
print(user_transition_score['buyer402'])
compare_profiles(user_transition_score['fraudster96'], user_transition_score['buyer402'], 6)
```

    {'login': 5.162147650850593, 'view_item': 33.24141988768278, 'sell_item': 0, 'buy_item': 49.60964047443681, 'end': 9.962896005337262, 'logout': 0, 'view_profile': 4.857980995127573, 'update_address': 5.169925001442313, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 0, 'update_email': 0}
    {'login': 5.162147650850593, 'view_item': 25.5340607556019, 'sell_item': 0, 'buy_item': 9.321928094887362, 'end': 9.962896005337262, 'logout': 9.271463027904375, 'view_profile': 0, 'update_address': 0, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 0, 'update_email': 0}





    {'dklp': 10.075853338434619, 'dklq': 8.9603374132886042, 'test': False}




```python
def remove_from_classification(candidate_name, behaviour_type_table):
    cleaneds = []
    empties = []
    for be, be_list in behaviour_type_table.items():
        if candidate_name in behaviour_type_table[be]:
            behaviour_type_table[be].remove(candidate_name)
            cleaneds.append(be)
        if len(behaviour_type_table[be]) == 0:
            empties.append(be)
    for e in empties:
        del behaviour_type_table[e]
            
    return cleaneds           
            

def classify_candidates(candidate_name, behaviour_type_table, score):
    potential_matching_type = {}
    passing_score = 0.85
    sample_size = 20
    small_size_adjustment = 2
    
    for be, be_list in behaviour_type_table.items():
        be_samples = random.sample(be_list, min(len(be_list), sample_size))

        post = 0.1 # this is the prior
        for idx in range(len(be_samples)):
            y = be_samples[idx]
            result = compare_profiles(score[candidate_name], score[y], 6)
            post = update_probability(post, flat_lookup, result['test'])
            
        if post >= passing_score * (min(small_size_adjustment,max(1,len(be_samples)))/small_size_adjustment):
            potential_matching_type[be] = post


    if len(potential_matching_type) == 0:
        
        new_class_name = max(0,len(list(behaviour_type_table.values()))) + 1
        return new_class_name
    else:
        return max(potential_matching_type, key=potential_matching_type.get)

def add_candidate_to_behaviour_type(candidate_name, matching_class, behaviour_type_table):  
    if matching_class not in behaviour_type_table.keys():
        behaviour_type_table[matching_class] = []

    if candidate_name not in behaviour_type_table[matching_class]:
        behaviour_type_table[matching_class].append(candidate_name)
        
    return candidate_name
    
def classify_users_in_list(unclassified_user_lists, behaviour_type_table, score):
    # select one user
    candidate_name = random.choice(unclassified_user_lists)
    if candidate_name:
        # classify user
        cleanup = remove_from_classification(candidate_name, behaviour_type_table)
        matching_class = classify_candidates(candidate_name, behaviour_type_table, score)

        # add the user to the proper type
        add_candidate_to_behaviour_type(candidate_name, matching_class, behaviour_type_table)
        unclassified_user_lists.remove(candidate_name)
    

```


```python
if random_seed:
    random.seed(random_seed)

behaviour_type_table = {}
unclassified_user_lists = random.sample(list(user_transition_score.keys()), len(list(user_transition_score.keys())))
```


```python
%%time
# while there are unclassified users
while len(unclassified_user_lists[:10]):
    classify_users_in_list(unclassified_user_lists, behaviour_type_table, user_transition_score)

for k in behaviour_type_table.keys():
    type_average = np.mean([sum(user_transition_score[x].values()) for x in behaviour_type_table[k]])
    print(k, type_average, len(behaviour_type_table[k]), cheat_lookup_all_users(behaviour_type_table[k]))
```

    1 116.343963314 238 {'merchant': 238}
    2 119.785949239 258 {'merchant': 258}
    3 87.4161188643 262 {'buyer': 262}
    4 86.5471276096 84 {'buyer': 84}
    5 180.252641893 93 {'buyer': 93}
    6 46.9206963748 4 {'merchant': 4}
    7 127.567312454 29 {'buyer': 29}
    8 101.680076675 2 {'buyer': 1, 'fraudster': 1}
    9 5.16214765085 1 {'merchant': 1}
    10 90.0648041663 2 {'fraudster': 2}
    11 172.650206702 3 {'merchant': 3}
    12 38.1590774773 6 {'buyer': 6}
    13 144.017241892 1 {'merchant': 1}
    14 32110.8953251 1 {'bot': 1}
    15 270.328618975 3 {'spammer': 3}
    16 48.9113281923 1 {'merchant': 1}
    17 8122.09717374 1 {'bot': 1}
    18 66.7039296268 3 {'attacker': 3}
    19 240.675677501 1 {'merchant': 1}
    20 81.345580945 3 {'attacker': 2, 'fraudster': 1}
    21 2922.64253355 1 {'bot': 1}
    22 24.7539745319 1 {'buyer': 1}
    23 176.545786547 1 {'buyer': 1}
    24 144.467124465 1 {'merchant': 1}
    CPU times: user 14 s, sys: 55.1 ms, total: 14.1 s
    Wall time: 14.2 s



```python
day1_data.loc[day1_data['user'].isin(behaviour_type_table[10])]['user'].unique()
```




    array(['fraudster96', 'fraudster86'], dtype=object)




```python
a = 'fraudster96'
b = 'fraudster86'

print(user_transition_score[a])
print(user_transition_score[b])
print('compare test', compare_profiles(user_transition_score[a], user_transition_score[b],6))
```

    {'login': 5.162147650850593, 'view_item': 33.24141988768278, 'sell_item': 0, 'buy_item': 49.60964047443681, 'end': 9.962896005337262, 'logout': 0, 'view_profile': 4.857980995127573, 'update_address': 5.169925001442313, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 0, 'update_email': 0}
    {'login': 5.162147650850593, 'view_item': 19.591826881094896, 'sell_item': 0, 'buy_item': 29.965784284662085, 'end': 9.962896005337262, 'logout': 0, 'view_profile': 4.857980995127573, 'update_address': 0, 'password_reset': 0, 'comment': 0, 'home': 0, 'bank_modify': 0, 'payment_modify': 2.584962500721156, 'update_email': 0}
    compare test {'test': True, 'dklp': 4.7391152106918337, 'dklq': 2.4413534715591183}


# Day 2, Let's see if we can find something

We are now Day 2. Surprisal values are based on a different day, but if the normal distribution is anything like Day 1, the surprisal value calculated should be still relevant.

Let's generate a new day of logs


```python
start_time = datetime(2019,1,2,0,0)
number_of_new_users = 20
existing_users = user_lists[:]

if random_seed:
    random.seed(random_seed + 1)

user_list_day2s = generate_userlist(number_of_new_users, existing_users)

if random_seed:
    random.seed(random_seed + 1)
day2_logs = generate_logs(user_list_day2s, start_time)

print(len(day2_logs), 'logs events generated for', len(user_list_day2s), 'users')

day2_data = transform_logs_to_pandas(day2_logs)
```

    18877 logs events generated for 1020 users



```python
user_transition_score_day2 = get_user_transition_score(day2_data, transition_surprisal, 'user', 'path', 'success')
```

This is where we use our log stream analyser. This is basically a state machine that keep an ongoing total of each users. 

The theory is that if a user only do normal actions, the sum of the surprisal value of all his actions should be fairly low (under ~10... but this is arbitrary). Anything over 10 would mean that a high number of unlikely actions were performed. Lets see if we can identify which users stands out.


```python
cumulative_score = [[v,sum(user_transition_score_day2[v].values())] for v in [k for k in list(user_transition_score_day2.keys())]]

df_cumulative_score = pd.DataFrame(cumulative_score, columns=['user', 'surprisal'])

avg = df_cumulative_score['surprisal'].mean()
std = df_cumulative_score['surprisal'].std()
df_cumulative_score['z'] = (df_cumulative_score['surprisal'] - avg) / std
```


```python
df_cumulative_score.loc[df_cumulative_score['z'] >= 2].sort_values(by=['surprisal'], ascending=False)
```


```python
if random_seed:
    random.seed(random_seed)

unclassified_user_lists = random.sample(list(user_transition_score_day2.keys()), len(list(user_transition_score_day2.keys())))
```


```python
%%time
# while there are unclassified users
while len(unclassified_user_lists):
    classify_users_in_list(unclassified_user_lists, behaviour_type_table, user_transition_score_day2)

for k in behaviour_type_table.keys():
    type_average = np.mean([sum(user_transition_score[x].values()) for x in behaviour_type_table[k]])
    print(k, type_average, len(behaviour_type_table[k]), cheat_lookup_all_users(behaviour_type_table[k]))
```

## Why these users are identified as outliers?

If we look at the Top 1 outlier, we can see each action performed, and the surprisal value of each action.

If we look at a normal user, we can see that the surprisal value assigned to each value is really low, only slightly contributing to give that user a high score.


```python
df_cumulative_score.sort_values(by=['surprisal'], ascending=False).tail()
```

# Day 3: Automatically Classifying Users

At one point, we still need to classify behaviours. Using Naive Bayes over the actions, we can score each users to the likely category they belong too.

Could we have done that without looking for the outliers? Yes, but we still need to identify which users is normal and which one is likely not.

At first, we need to calculate the probability distribution of each actions for each categories.

We can now use this probability distribution over a log stream