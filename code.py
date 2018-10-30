
# coding: utf-8

# NEW YORK CITY TAXI PREDICTION

# Importing required libraries:

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats.stats import pearsonr
import datetime
from dateutil import tz
from haversine import haversine
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# The training dataset provided contains ~55M entries, which is too much for my 8gb machine too handle. Thus I sampled the dataset, taking only 5M entries for training purposes.

# In[2]:


fp = "train.csv"
noOfLines = sum(1 for l in open(fp))
rowsSkipped = [x for x in range(1,noOfLines) if x%11!=0]


# In[3]:


train_df = pd.read_csv(fp, skiprows = rowsSkipped)


# In[4]:


train_df.describe()


# TASK 1:  DATA CLEANING
# Data cleaning done:
#  - 'fare_amount' contained negative fares (as observed from min in the describe table). However, on analysing the data, I realised that some of the negative fares were still valid as they had sensible data in all the other rows.
#      - Deleted values below minimum fare (-2.5 < x < 2.5)
#      - Converted all negative values to positive
#  - Missing values in 'dropoff_latitude' and 'dropoff_longitude'(as observed from count in the describe table). 
#      - Drop NA rows
#  - The Earth's latitude must lie between -90 and 90 and longitude must lie between -180 and 180. All latitude and longitude columns had some values greater than that amount on both sides (as observed from min and max in the describe table).
#      - Dropped rows where (pickup_longitude, pickup_latitude, dropoff_longitude & dropoff_latitude > 90) and (pickup_longitude, pickup_latitude, dropoff_longitude & dropoff_latitude < -90)
#  - New York City is nowhere near 0 latitude and 0 longitude, yet there are several rows with latitude and longitude degrees = 0 (as observed on analysing the whole training dataset).
#      - Drop rows where pickup_longitude, pickup_latitude, dropoff_longitude & dropoff_latitude = 0
#  - Tried to come up with a way to deal with outlier distance values. Therefore, I defined a bounding box using coordinates obtained from Google Maps for where it was feasible for the taxi to travel. Included a majority of New York state, Conneticut, Pennsylvania, New Jersey and upper Maryland. Beyond that signifies outliers. Note that the right latitude range is smaller as it signifies water i.e. Atlantic Ocean.
#      - The centre point of New York is estimated to be : 
#      - Dropped all values pickup_longitude, pickup_latitude > 44.05 and < 36.05, dropoff_longitude , dropoff_latitude > -71.50 and < 80.01.
#  - Observed that passenger_count table had few outliers which skewed the graphs drawn. 
#      - Dropped all passenger count > 100
#  
#  Further data cleaning was conducted after calculating the Euclidean distance between the two points.
#  

# In[5]:


train_df = train_df[(train_df['fare_amount'] >= 2.50) | (train_df['fare_amount'] <= -2.50)  ]
res = train_df.apply(lambda row: abs(row['fare_amount']), axis = 1)
train_df = train_df.assign(fare_amount = res.values)


# In[6]:


train_df = train_df.dropna()


# In[7]:


train_df = train_df [(train_df['pickup_longitude'] >= -90) & (train_df['pickup_longitude'] <= 90) ]


# In[8]:


train_df = train_df [(train_df['pickup_latitude'] >= -90) & (train_df['pickup_latitude'] <= 90) ]


# In[9]:


train_df = train_df [(train_df['dropoff_longitude'] >= -90) & (train_df['dropoff_longitude'] <= 90) ]


# In[10]:


train_df = train_df [(train_df['dropoff_latitude'] >= -90) & (train_df['dropoff_latitude'] <= 90) ]


# In[14]:


train_df = train_df [(train_df['pickup_longitude'] !=0) & (train_df['pickup_latitude'] !=0)]


# In[18]:


train_df = train_df[(train_df['pickup_latitude'] <= 43.05) & (train_df['dropoff_latitude'] <= 43.05 )]


# In[20]:


train_df = train_df[(train_df['pickup_latitude'] >= 39.05) & (train_df['dropoff_latitude'] >= 39.05 )]


# In[22]:


train_df = train_df[(train_df['pickup_longitude'] <= -71.50) & (train_df['dropoff_longitude'] <= -71.50 )]


# In[24]:


train_df = train_df[(train_df['pickup_longitude'] >= -75.01) & (train_df['dropoff_longitude'] >= -75.01 )]


# In[26]:


train_df = train_df[train_df['passenger_count'] <8]


# TASK 2: PEARSON CORRELATION
#  

# 1) Euclidean vs taxi fare:
#     - First we calculate the Euclidean distance. 
#     - Convert the lat/long degrees to cartesian coordinates x & y where x = R x cos(lat) x cos(long) & y = R x cos(lat) x sin(long)
#     - Data cleaning based on Euclidean distance
#     - Then we calculate Pearson for euclidean vs taxi fare

# In[30]:


def eucl(p_lat, p_long,d_lat,d_long):
    x1 = 6371*math.cos(math.radians(p_lat))*math.cos(math.radians(p_long))
    y1 = 6371*math.cos(math.radians(p_lat))*math.sin(math.radians(p_long))
    x2 = 6371*math.cos(math.radians(d_lat))*math.cos(math.radians(d_long))
    y2 = 6371*math.cos(math.radians(d_lat))*math.sin(math.radians(d_long))
    return (math.sqrt((x2-x1)**2 + (y2-y1)**2)*0.621371) 


# In[31]:


res = train_df.apply(lambda row: eucl(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude'] ), axis = 1 )
train_df = train_df.assign(euclidean = res.values)


# Some data cleaning based on Euclidean distance:
#     - Don't pay if you don't travel any distance by the cab (just getting into the cab and getting out immediately)
#         - Remove all distances where 'euclidean' = 0 
#     - Highly improbable that cab will travel a distance > 500 miles
#         - Removed all 'euclidean' > 500
#     - Analysed a graph plotting 'euclidean' vs 'fare_amount'. Identified that data was mostly noisy for 'euclidean' > 150 (low 'fare_amount' for high 'euclidean' - highly improbable) and 'fare_amount' > 350 (low 'euclidean' for high 'fare_amount')
#         - Removed all 'euclidean' > 150 
#         - Removed all 'fare_amount > 350'

# In[32]:


train_df = train_df[(train_df['euclidean'] > 0)]


# In[36]:


train_df = train_df[(train_df['euclidean'] <= 500) ]


# In[38]:


sb.relplot(x="euclidean", y="fare_amount", data=train_df);
plt.show()


# From plot we try to cut off values. 
#     - We see that for euclidean > 150 fare amount is close to zero only which is impossible intuitively. 
#     - We see that for distance close to 0, fare_amount reaches 500, we try to cut it off at 350

# In[39]:


train_df = train_df[(train_df['euclidean'] <= 150) ]
train_df = train_df[(train_df['fare_amount'] <= 350) ]
len(train_df)


# In[41]:


print (pearsonr(train_df['euclidean'] , train_df['fare_amount']))    


# 2) Time of day vs Distance traveled
#     - Given column 'pickup_datetime' is a string, so we have to convert it into datetime format
#     - Extract time of day (Not in terms of hour but in terms of seconds) 
#     - Compare with Euclidean distance

# In[42]:


import datetime
res = train_df.apply(lambda row: datetime.datetime.strptime(row['pickup_datetime'], "%Y-%m-%d %H:%M:%S %Z"), axis=1)
train_df = train_df.assign(pickup_datetime = res.values)


# In[43]:


time = train_df.apply(lambda row: ((row['pickup_datetime'].hour*60*60) + (row['pickup_datetime'].minute*60) + (row['pickup_datetime'].second)), axis = 1)
train_df = train_df.assign(time = time.values)


# In[44]:


print (pearsonr(train_df['euclidean'] , train_df['time']))


# 3)  Time of day vs Taxi fare
#     - Compare as above

# In[45]:


print (pearsonr(train_df['fare_amount'] , train_df['time']))


# The Pearson correlation of the above three comparisons are:
#     1) Euclidean distance of the ride vs Taxi fare: 0.7823848626099872 (Highly correlated)
#     2) Time of day vs Distance travelled: -0.025729621971280942 (Not correlated)
#     3) Time of day vs Fare Amount: -0.017458907381592163 (Not correlated)
# Thus, the highest correlation is between Euclidean distance vs Taxi Fare

# TASK 3: VISUALIZING RELATION BETWEEN ABOVE PAIRS OF VARIABLES

# 1) Euclidean distance vs Fare amount

# In[46]:


sb.relplot(x="euclidean", y="fare_amount", data=train_df);
plt.show()


# 2) Time of Day vs Euclidean distance

# In[47]:


sb.relplot(x="time", y="euclidean", data=train_df);
plt.show()


# 3) Time of Day vs Fare Amount

# In[48]:


sb.relplot(x="time", y="fare_amount", data=train_df);
plt.show()


# Comment about above visualisations:
# 
# 1) Euclidean distance vs Taxi Fare:
#         - The high correlation between Euclidean distance and Taxi fare is obvious from the graph. A clear linear relation is seen in the graph. 
#         - The taxi fare increases as the euclidean distance increases, thus explaining the positive correlation between the two attributes.
#         
# 2) Time of day vs Euclidean distance:
#         - We can see that the data points are randomly scattered throughout the graph. Thus, there is no true relationship between the time of the day and the Euclidean distance, explaining why there is no correlation between the two attributes (correlation is very small and insignificant).
#         - There are two clear horizontal lines in the visualization. It is likely that these lines signify trips from airports which have a fixed rate and high amount of traffic. We observe that most people arrive or depart from the airport in a time period from
#         
# 3) Time of day vs Fare amount
#         - We can see that the data points are randomly scattered throughout the graph. Thus, there is no true relationship between the time of the day and the Euclidean distance, explaining why there is no correlation between the two attributes (correlation is very small and insignificant).

# We create a few additional features to add to the model:
# - Haversine Distance : haversine
#     - Haversine distance or 'as the crow flies' determines the great-circle distance between two points on a sphere given their longitudes and latitudes. Since it takes the spherical shape of the Earth into account while estimating distance, it is taken as an attribute to hopefully get a more accurate distance measure.
# - Rush hour : is_rush_hour
#      - There is a 1\$ surcharge on the taxi fare during rush hour. Rush hour is defined as 4pm to 8pm on weekdays. Therefore, we create a boolean attribute that indicates whether the trip occured in the rush hour period or not.
# - Night trip charge : is\_night
#      - There is a 0.50\$ surcharge on the taxi fare for a night trip. A night trip is defined as a trip that takes place from 8 pm to 6 am. Therefore, we create a boolean attribute that indicates whether the trip is a night trip or not.
# - JFK airport: is\_JFK
#      - There is a flat fare of \$52 for all trips originating and culminating at JFK airport. This flat price was \$45 before 2012. In addition, there is a flat fare of \$4.50 for all trips that occur during rush hour. Therefore, I defined a four-point bounding box manually using Google Maps to locate coordinates. The coordinates are:  upper = 40.664, lower = 40.626, left = -74.192, right = -74.163. So, we create a boolean attribute that indicates if the origin/destination is JFK Airport.
# - Newark airport: is\_newark
#      - There is a \$17.50 surcharge on taxi fares to and from Newark Airport in addition to the metered fare. Therefore, I defined a four-point bounding box manually using Google Maps to locate coordinates. The coordinates are: upper = 40.717, lower = 40.665, left = -74.192 and right = -74.163. This is used to create a boolean attribute to indicate if the origin/destination is Newark Airport. 
# - Estimated fare in miles : est_fare
#      - The fare of a taxi is a function of the waiting time, the distance in miles travelled by the car on its journey and other surcharges. Since we have not been given the dropoff time, it is difficult to estimate the total duration of the journey. However, we try to define the fare based on the surcharges and distance as of now. The base fare is \$2.50.      
# - Grid distance: grid\_distance
#     - Manhattan has a grid-like structure so I expect a more accurate measure of distance would be the Manhattan distance such that it is the sum of the absolute distance between two sets of coordinates. I look to experiment with different distance measures to see which one works best for our problem.
# - Day of the week, Hour of the day and Year: day\_of\_week, hour\_of\_day and year
#     - I decided to extract data from the timestamp that intuitively felt the most important for prediction. The first two attributes relate to traffic - there could be more traffic during certain hours of the day and certain days of the week resulting in a higher fare. I added the year attribute as research revealed that there was a change (increase) in pricing of the taxi fares in 2012.
#     

# 1) Haversine distance

# In[49]:


res = train_df.apply(lambda row: haversine((row['pickup_latitude'] , row['pickup_longitude']),(row['dropoff_latitude'], row['dropoff_longitude']), miles = True), axis=1)
train_df = train_df.assign(haversine = res.values)


# In[50]:


train_df.head()


# 2) Rush Hour
#     Rush hour is defined as 4pm to 8pm on weekdays

# In[51]:


def rush_hour(hr,wkd):
    if ((hr >= 16) & (hr <20) & (wkd<5)):
        return 1
    return 0


# In[52]:


res = train_df.apply(lambda row: rush_hour(row['pickup_datetime'].hour, row['pickup_datetime'].weekday()), axis = 1)
train_df = train_df.assign(is_rush_hour = res.values)


# In[53]:


train_df.head()


# 3) Night charge
#     Night is defined as 8pm to 6am 

# In[54]:


def is_night(hr):
    if ((hr>= 20) & (hr <24)):
        return 1
    elif ((hr>= 0) & (hr <6)):
        return 1
    else: 
        return 0


# In[55]:


res = train_df.apply(lambda row: is_night(row['pickup_datetime'].hour), axis = 1)
train_df = train_df.assign(is_night = res.values)


# In[56]:


train_df.head()


# 4) JFK: By looking at Google maps, we define JFK in a bounded box where upper = 40.664, lower = 40.626, left = -74.192, right = -74.163

# In[57]:


def is_jfk(p_lat, p_long, d_lat, d_long):
    if((p_lat <= 40.664) & (p_lat >= 40.626)):
        if((p_long >= -73.82) & (p_long <= -73.77)):
            return 1
    
    if ((d_lat <= 40.664) & (d_lat >= 40.626)):
        if((d_long >= -73.82) & (d_long <=-73.77)):
            return 1
    
    return 0
                    


# In[58]:


res = train_df.apply(lambda row: is_jfk(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis = 1)
train_df = train_df.assign(is_jfk = res.values)


# In[59]:


train_df.loc[train_df['is_jfk'] == 1]


# 5) Newark Airport: By looking at Google maps, we define Newark airport in a bounded box where upper = 40.717, lower = 40.665, left = -74.192 and right = -74.163

# In[60]:


def is_newark(p_lat, p_long, d_lat, d_long):
    if((p_lat <= 40.717) & (p_lat >= 40.665)):
        if((p_long >= -74.192) & (p_long <= -74.163)):
            return 1
    
    if ((d_lat <= 40.717) & (d_lat >= 40.665)):
        if((d_long >= -74.192) & (d_long <=-74.163)):
            return 1
    
    return 0


# In[61]:


res = train_df.apply(lambda row: is_newark(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis = 1)
train_df = train_df.assign(is_newark = res.values)


# In[62]:


train_df.loc[train_df['is_newark'] == 1]


# Estimated fare: We try to estimate a fare for the taxi (assuming no waiting time) and see the results.

# In[63]:


def est_fare(hav, rush, night, jfk, nwrk, year ):
    if (jfk):
        if (year >= 2012):
            if(rush):
                return 56.5
            else:
                return 52
        else:
            if(rush):
                return 49.5
            else:
                return 45
    
    else:
        fare = 2.5
        fare = fare + 0.5*(hav/0.2)
        if (rush):
            fare = fare + 1
        if (night):
            fare = fare + 0.5
        if (nwrk):
            fare = fare + 17.5
        return fare


# In[64]:


res = train_df.apply(lambda row: est_fare(row['haversine'], row['is_rush_hour'], row['is_night'], row['is_jfk'], row['is_newark'], row['pickup_datetime'].year), axis = 1)
train_df = train_df.assign(est_fare = res.values)


# In[65]:


train_df.head()


# In[66]:


def grid_dist(p_lat, p_long,d_lat,d_long):
    x1 = 6371*math.cos(math.radians(p_lat))*math.cos(math.radians(p_long))
    y1 = 6371*math.cos(math.radians(p_lat))*math.sin(math.radians(p_long))
    x2 = 6371*math.cos(math.radians(d_lat))*math.cos(math.radians(d_long))
    y2 = 6371*math.cos(math.radians(d_lat))*math.sin(math.radians(d_long))
    return ((abs(x1-x2) + abs(y1-y2))*0.621371) 


# In[67]:


res = train_df.apply(lambda row: grid_dist(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude'] ), axis = 1 )
train_df = train_df.assign(grid_dist = res.values)


# In[68]:


res = train_df.apply(lambda row: row['pickup_datetime'].weekday(), axis = 1 )
train_df = train_df.assign(day_of_week = res.values)


# In[69]:


res = train_df.apply(lambda row: row['pickup_datetime'].hour, axis = 1 )
train_df = train_df.assign(hour_of_day = res.values)


# In[70]:


res = train_df.apply(lambda row: row['pickup_datetime'].year, axis=1)
train_df = train_df.assign(year = res.values)


# In[71]:


train_df.head()


# Visualizing few features:
#     

# In[72]:


sb.relplot(x="day_of_week", y="fare_amount", data=train_df);
plt.show()


# In[73]:


print (pearsonr(train_df['day_of_week'] , train_df['fare_amount']))


# In[74]:


sb.relplot(x="hour_of_day", y="fare_amount", data=train_df);
plt.show()


# In[75]:


print (pearsonr(train_df['hour_of_day'] , train_df['fare_amount']))


# In[76]:


sb.relplot(x="year", y="fare_amount", data=train_df);
plt.show()


# In[77]:


print (pearsonr(train_df['year'] , train_df['fare_amount']))


# In[78]:


sb.relplot(x="dropoff_longitude", y="fare_amount", data=train_df);
plt.show()


# In[79]:


print (pearsonr(train_df['dropoff_longitude'] , train_df['fare_amount']))


# In[80]:


Y = train_df.apply(lambda row: row['fare_amount'], axis=1)
train_df, test_df, train_Y, test_Y = model_selection.train_test_split(train_df, Y, test_size=0.33, random_state=42)
train_df = train_df.drop(columns=['pickup_datetime','fare_amount'])
test_df = test_df.drop(columns=['pickup_datetime','fare_amount'])


# In[81]:


lr = linear_model.LinearRegression()


# In[82]:


le = LabelEncoder()
le.fit(train_df['key'].astype(str))
train_df['key'] = le.transform(train_df['key'].astype(str))
le.fit(test_df['key'].astype(str))
test_df['key'] = le.transform(test_df['key'].astype(str))
lr.fit(train_df, train_Y)
predictions = lr.predict(test_df)
mse = np.mean((test_Y-lr.predict(test_df))**2)


# In[83]:


math.sqrt(mse)


# In[84]:


lr.coef_


# In[85]:


rf = RandomForestRegressor(n_estimators = 3, random_state = 42)
rf.fit(train_df, train_Y)


# In[86]:


mse = np.mean((test_Y-rf.predict(test_df))**2)


# In[87]:


print (mse)


# In[88]:


rf.feature_importances_


# In[ ]:


train_df = train_df.drop(columns=['passenger_count','is_rush_hour','is_night','is_jfk','is_newark','grid_dist','day_of_week','hour_of_day','year'])
test_df = test_df.drop(columns=['passenger_count','is_rush_hour','is_night','is_jfk','is_newark','grid_dist','day_of_week','hour_of_day','year'])


# In[ ]:


rf = RandomForestRegressor(n_estimators = 3, random_state = 42)
rf.fit(train_df, train_Y)


# In[90]:


mse = np.mean((test_Y-rf.predict(test_df))**2)


# In[91]:


print(mse)


# In[92]:


test_df = pd.read_csv("test.csv")


# In[ ]:


test_df.describe()


# In[93]:


res = test_df.apply(lambda row: eucl(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude'] ), axis = 1 )
test_df = test_df.assign(euclidean = res.values)


# In[94]:


res = test_df.apply(lambda row: datetime.datetime.strptime(row['pickup_datetime'], "%Y-%m-%d %H:%M:%S %Z"), axis=1)
test_df = test_df.assign(pickup_datetime = res.values)
time = test_df.apply(lambda row: ((row['pickup_datetime'].hour*60*60) + (row['pickup_datetime'].minute*60) + (row['pickup_datetime'].second)), axis = 1)
test_df = test_df.assign(time = time.values)
res = test_df.apply(lambda row: haversine((row['pickup_latitude'] , row['pickup_longitude']),(row['dropoff_latitude'], row['dropoff_longitude']), miles = True), axis=1)
test_df = test_df.assign(haversine = res.values)
res = test_df.apply(lambda row: rush_hour(row['pickup_datetime'].hour, row['pickup_datetime'].weekday()), axis = 1)
test_df = test_df.assign(is_rush_hour = res.values)
res = test_df.apply(lambda row: is_night(row['pickup_datetime'].hour), axis = 1)
test_df = test_df.assign(is_night = res.values)
res = test_df.apply(lambda row: is_jfk(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis = 1)
test_df = test_df.assign(is_jfk = res.values)
res = test_df.apply(lambda row: is_newark(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis = 1)
test_df = test_df.assign(is_newark = res.values)
res = test_df.apply(lambda row: est_fare(row['haversine'], row['is_rush_hour'], row['is_night'], row['is_jfk'], row['is_newark'], row['pickup_datetime'].year), axis = 1)
test_df = test_df.assign(est_fare = res.values)


# In[95]:


test_df = test_df.drop(columns=['pickup_datetime','passenger_count','is_rush_hour','is_night','is_jfk','is_newark'])


# In[96]:


test_df.describe()


# In[ ]:


keys = test_df['key']
le.fit(test_df['key'].astype(str))
test_df['key'] = le.transform(test_df['key'].astype(str))
predictions = rf.predict(test_df)


# In[ ]:


predic = pd.DataFrame({'key': keys, 'fare_amount': predictions})
predic.to_csv('submission.csv', index = False)


# External datasets: 
#     - One thing that was difficult to realize as a feature was the duration of the journey. Since the meter hikes up the price of the fare by $0.50 every minute, the duration of the journey would serve as an important feature for the model. One possible way of implementing this would be by finding data that already has the duration of the journey as an attribute and using imputation (with the help of regression possibly) to generate data points for duration of journey for our training dataset. The given training data is a subset of the data provided by the NYC TLC (Taxi and Limousine Commission) that has recorded taxi trip data from 2009 to 2018. (Source: http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml). This data has additional features, one of which is 'dropoff_datetime'. Thus, we can calculate the journey duration using this additional data rows and impute the values for the ones in the training set.
#     - The coordinates for the airports were retrieved manually by defining a four-point bounding box via Google Maps. Thus, it is an approximation of airport area. I found an online dataset that defined a polygon for airports in NY (Airport Polygon: https://data.cityofnewyork.us/City-Government/Airport-Polygon/xfhz-rhsk) that will give an accurate representation of drop-off/pickup points actually in the airport.
#     - I originally pre-processed the data by defining a bounding box of the area where the taxi can travel. We could tighten the box, but the coordinates were difficult to retrieve manually via Google Maps. I found a dataset available online that defines the boundaries of the 5 boroughs via coordinate points (Borough Boundaries: https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm). We can use this to pre-process the data so as to tighten it such that we cover only the main New York City area. This also helps remove coordinates from the dataset that are in the middle of the sea, that could not be removed via previous preprocessing methods.  
#     - All of the distance measures we use are not the most accurate distance measures. Obstacles that may block the path of a vehicle are not accounted for. The most accurate measure of distance would be to use Google Map data to find the distance between two sets of coordinates. This however was not possible to implement as usage of the API required payment.
#     
