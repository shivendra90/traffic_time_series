https://www.kaggle.com/rohith203/traffic-volume-dataset#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:00:10 2019

@author: Shivendra

For ease of use, the modules used by this analysis
are included in the load_libs.py file. Since it is
always more efficient to load all the modules through
one-liners, it is always advisable to load that file.
"""

"""
Section: 1
Load Libraries and data
"""
# exec(open("load_libs.py").read())
# Use the above command to load the libraries in one click


from statsmodels.tsa.statespace.sarimax import SARIMAX
from numpy import log
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor, XGBRFRegressor
from yellowbrick.regressor import residuals_plot, prediction_error
from yellowbrick.features import rank2d, rank1d
from yellowbrick.model_selection import RFECV, ValidationCurve, LearningCurve
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDRegressor, PassiveAggressiveRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR, NuSVR
from warnings import filterwarnings
from matplotlib.pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from seaborn import catplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas_profiling
import statsmodels.api as sm
import sklearn.metrics as metrics
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
register_matplotlib_converters()
filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
# rcParams['figure.figsize'] = 10, 8
plt.ion()
np.random.seed(1000)
print("\nEnvironment is ready.")

main_data = dd.read_csv("Train.csv")
test_set = pd.read_csv("Test.csv")  # 14454 rows
submission = pd.DataFrame({"date_time": test_set['date_time'],
                           "traffic_volume": np.nan})

"""
Section: 2
Exploratory Analysis
"""
print(main_data.columns)

# Check duplicate time_stamps
print(len(main_data) == main_data.date_time.nunique())  # Returns True
print(len(test_set) == test_set.date_time.nunique())  # True again

main_data.traffic_volume.describe()

# A rough plot
main_data.traffic_volume.plt(alpha=0.4)

print("Percent of NA values: \n{0}".format((main_data.isna().sum()) / 100))
print("Percent of NA values: \n{0}".format((test_set.isna().sum()) / 100))

# Value counts
main_data.weather_type.value_counts()
main_data.is_holiday.value_counts()
main_data.dew_point.value_counts()
main_data.visibility_in_miles.value_counts()

# Frequency plots
holiday_count = main_data.is_holiday.value_counts()
weather_count = main_data.weather_type.value_counts()

sns.barplot(holiday_count.index, holiday_count.values)
plt.title("Frequency Distribution (holidays)")
plt.show()

sns.barplot(weather_count.index, weather_count.values)
plt.title("Frequency Distribution (weather types)")
plt.show()

# Resample data to month start and show mean of traffic volume
traffic_resampled = main_data["traffic_volume"].resample("MS").mean()

traffic_resampled.plot()
plt.xlabel("Year")
plt.ylabel("Traffic Freq.")
plt.title("Traffic Volume 2012-17")

decomposition = sm.tsa.seasonal_decompose(traffic_resampled, model='additive')

decomposition.plot()
# Clean up
del weather_count, holiday_count

"""
Section: 3
Preprocessing/Data Preparation
"""
main_data["date_time"] = pd.to_datetime(main_data.date_time,
                                        format="%Y-%m-%d %H:%M:%S")
main_data = main_data.set_index("date_time")

plt.plot("traffic_volume", data=main_data, alpha=0.3, marker='.')

# Work on missing dates
missing_range_1 = pd.date_range(
    '2013-10-27 00:00:00', '2013-11-07 05:00:00', freq="H")
baseline_nov_2012 = main_data.loc['2012-10-27':'2012-11-06']

gen_data = pd.DataFrame({'date_time': missing_range_1,
                         # Since no Day is a holiday
                         'is_holiday': baseline_nov_2012['is_holiday'],
                         'air_pollution_index': baseline_nov_2012['air_pollution_index'],
                         'humidity': baseline_nov_2012['humidity'],
                         'wind_speed': baseline_nov_2012['wind_speed'],
                         'wind_direction': baseline_nov_2012['wind_direction'],
                         'visibility_in_miles': baseline_nov_2012['visibility_in_miles'],
                         'dew_point': baseline_nov_2012['dew_point'],
                         'temperature': baseline_nov_2012['temperature'],
                         'rain_p_h': baseline_nov_2012['rain_p_h'],
                         'snow_p_h': baseline_nov_2012['snow_p_h'],
                         'clouds_all': baseline_nov_2012['clouds_all'],
                         'weather_type': baseline_nov_2012['weather_type'],
                         'weather_description': baseline_nov_2012['weather_description'],
                         'traffic_volume': baseline_nov_2012['traffic_volume']})

gen_data = gen_data.set_index("date_time")

# Join main_data and gen_data
main_data = main_data.append(gen_data, sort=True)

# Work with the most demanding gap
missing_range_2 = pd.date_range(
    "2014-08-08 03:00:00", "2015-06-24 10:00:00", freq="H")
len(missing_range_2)  # 7688 rows

baseline2013_2014 = main_data.loc['2013-08-01':'2014-07-14']
len(baseline2013_2014)  # 7688 rows

gen_data = pd.DataFrame({'date_time': missing_range_2,
                         'is_holiday': baseline2013_2014['is_holiday'],
                         'air_pollution_index': baseline2013_2014['air_pollution_index'],
                         'humidity': baseline2013_2014['humidity'],
                         'wind_speed': baseline2013_2014['wind_speed'],
                         'wind_direction': baseline2013_2014['wind_direction'],
                         'visibility_in_miles': baseline2013_2014['visibility_in_miles'],
                         'dew_point': baseline2013_2014['dew_point'],
                         'temperature': baseline2013_2014['temperature'],
                         'rain_p_h': baseline2013_2014['rain_p_h'],
                         'snow_p_h': baseline2013_2014['snow_p_h'],
                         'clouds_all': baseline2013_2014['clouds_all'],
                         'weather_type': baseline2013_2014['weather_type'],
                         'weather_description': baseline2013_2014['weather_description'],
                         'traffic_volume': baseline2013_2014['traffic_volume']})

gen_data = gen_data.set_index("date_time")

main_data = main_data.append(gen_data, sort=True)

# Clear out duplicate time stamp indices
del baseline2013_2014, baseline_nov_2012, gen_data, missing_range_1, missing_range_2
main_data = main_data.sort_index()  # 41708 rows

main_data = main_data.loc[~main_data.index.duplicated(keep='last')]

target = main_data["traffic_volume"].copy()  # Keep this safe
main_data.drop("traffic_volume", axis=1, inplace=True)

main_data["Year"] = main_data.index.year
main_data["Month"] = main_data.index.month
main_data["Day"] = main_data.index.day
main_data["Hour"] = main_data.index.hour

# Work with categorical data
print(main_data.dtypes)
# Subset for better visualisation
obj_df = main_data.select_dtypes(include=["object"]).copy()

label_dict = {
    "is_holiday": {"None": 0, "New Years Day": 1, "Thanksgiving Day": 2, "Christmas Day": 3, "Columbus Day": 4,
                   "Labor Day": 5, "Veterans Day": 6, "Washingtons Birthday": 7, "Memorial Day": 8,
                   "Martin Luther King Jr Day": 9,
                   "State Fair": 10, "Independence Day": 11},
    "weather_type": {"Clear": 0, "Clouds": 1, "Mist": 2, "Rain": 3, "Snow": 4, "Drizzle": 5, "Haze": 6, "Fog": 7,
                     "Thunderstorm": 8,
                     "Smoke": 9, "Squall": 10},
    "weather_description": {"sky is clear": 0, "Sky is Clear": 0, "mist": 1, "broken clouds": 2, "scattered clouds": 2,
                            "light intensity shower rain": 3,
                            "light rain": 3, "light intensity rain": 3, "light intensity drizzle": 3, "drizzle": 3,
                            "shower drizzle": 3, "heavy intensity rain": 4, "heavy intensity drizzle": 4,
                            "haze": 5, "smoke": 5, "thunderstorm with light rain": 6,
                            "thunderstorm with light drizzle": 6, "thunderstorm with drizzle": 6, "freezing rain": 7,
                            "sleet": 7, "shower snow": 7, "light rain and snow": 7, "light shower snow": 7,
                            "overcast clouds": 8, "few clouds": 9, "light snow": 10, "moderate rain": 11,
                            "heavy snow": 12, "proximity thunderstorm": 13, "snow": 14,
                            "thunderstorm": 15, "proximity shower rain": 16, "thunderstorm with heavy rain": 17,
                            "proximity thunderstorm with rain": 18, "thunderstorm with rain": 19,
                            "very heavy rain": 20, "proximity thunderstorm with drizzle": 21, "fog": 22, "SQUALLS": 23}}

obj_df.replace(label_dict, inplace=True)
main_data.replace(label_dict, inplace=True)

# Clean up holidays for newly inserted rows

# Holiday clean up
# First fill wrong entries with zeros
main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 12) & (main_data.index.day == 12),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2017) & (main_data.index.month == 1) & (main_data.index.day == 2),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 11) & (main_data.index.day == 3),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 12) & (main_data.index.day == 4),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 12) & (main_data.index.day == 26),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 10) & (main_data.index.day == 3),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 9) & (main_data.index.day == 4),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 10) & (main_data.index.day == 17),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 1) & (main_data.index.day == 30),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 5) & (main_data.index.day == 8),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 1) & (main_data.index.day == 1),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 7) & (main_data.index.day == 3),
              ['is_holiday']] = 0

main_data.loc[(main_data.index.year == 2012) & (main_data.index.month == 11) & (main_data.index.day == 12),
              ['is_holiday']] = 0

# Fill holidays with common dates lke New Year's
main_data.loc[(main_data.index.month == 1) & (main_data.index.day == 1),
              ['is_holiday']] = 1  # New Year

main_data.loc[(main_data.index.month == 12) & (main_data.index.day == 25),
              ['is_holiday']] = 3  # Xmas

main_data.loc[(main_data.index.month == 11) & (main_data.index.day == 11),
              ['is_holiday']] = 6  # Veterans' Day

main_data.loc[(main_data.index.month == 7) & (main_data.index.day == 4),
              ['is_holiday']] = 11  # Independence Day

# For holidays with differing days
# Thanksgiving
main_data.loc[(main_data.index.year == 2012) & (main_data.index.month == 11) & (main_data.index.day == 22),
              ['is_holiday']] = 2

main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 11) & (main_data.index.day == 28),
              ['is_holiday']] = 2

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 11) & (main_data.index.day == 27),
              ['is_holiday']] = 2

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 11) & (main_data.index.day == 26),
              ['is_holiday']] = 2

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 11) & (main_data.index.day == 24),
              ['is_holiday']] = 2

# Columbus
main_data.loc[(main_data.index.year == 2012) & (main_data.index.month == 10) & (main_data.index.day == 8),
              ['is_holiday']] = 4

main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 10) & (main_data.index.day == 14),
              ['is_holiday']] = 4

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 10) & (main_data.index.day == 13),
              ['is_holiday']] = 4

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 10) & (main_data.index.day == 12),
              ['is_holiday']] = 4

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 10) & (main_data.index.day == 10),
              ['is_holiday']] = 4

# Labour Day
main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 9) & (main_data.index.day == 2),
              ['is_holiday']] = 5

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 9) & (main_data.index.day == 1),
              ['is_holiday']] = 5

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 9) & (main_data.index.day == 7),
              ['is_holiday']] = 5

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 9) & (main_data.index.day == 5),
              ['is_holiday']] = 5

# Washington's B'day
main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 2) & (main_data.index.day == 18),
              ['is_holiday']] = 7

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 2) & (main_data.index.day == 17),
              ['is_holiday']] = 7

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 2) & (main_data.index.day == 16),
              ['is_holiday']] = 7

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 2) & (main_data.index.day == 15),
              ['is_holiday']] = 7

# Memorial Day
main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 5) & (main_data.index.day == 27),
              ['is_holiday']] = 8

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 5) & (main_data.index.day == 26),
              ['is_holiday']] = 8

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 5) & (main_data.index.day == 25),
              ['is_holiday']] = 8

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 5) & (main_data.index.day == 30),
              ['is_holiday']] = 8

# Martin Luther Day
main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 1) & (main_data.index.day == 21),
              ['is_holiday']] = 9

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 1) & (main_data.index.day == 20),
              ['is_holiday']] = 9

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 1) & (main_data.index.day == 19),
              ['is_holiday']] = 9

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 1) & (main_data.index.day == 18),
              ['is_holiday']] = 9

main_data.loc[(main_data.index.year == 2017) & (main_data.index.month == 1) & (main_data.index.day == 16),
              ['is_holiday']] = 9

# State Fair
main_data.loc[(main_data.index.year == 2013) & (main_data.index.month == 8) & (main_data.index.day == 22),
              ['is_holiday']] = 10

main_data.loc[(main_data.index.year == 2014) & (main_data.index.month == 8) & (main_data.index.day == 28),
              ['is_holiday']] = 10

main_data.loc[(main_data.index.year == 2015) & (main_data.index.month == 8) & (main_data.index.day == 27),
              ['is_holiday']] = 10

main_data.loc[(main_data.index.year == 2016) & (main_data.index.month == 8) & (main_data.index.day == 25),
              ['is_holiday']] = 10

# Some plotting
cols_to_plot = ["air_pollution_index", "dew_point",
                "humidity", "temperature", "wind_speed"]

fig, axes = plt.subplots(nrows=len(cols_to_plot))
for i, var in enumerate(cols_to_plot):
    main_data[var].plt.density(ax=axes[i])

main_data['traffic_volume'] = target
group_df = main_data.resample("D").interpolate()[::7]

fig, (axes_1, axes_2, axes_3) = plt.subplots(nrows=3, ncols=1)
axes_1.plt(group_df['traffic_volume'].groupby([group_df.index.week]).mean())
axes_1.set_title("Traffic Volume")

axes_2.plt(group_df['dew_point'], color='orange', label="Dew point")
axes_2.set_title("Dew Point & Visibility (miles)")

axes_3.plt(group_df['visibility_in_miles'],
           color='green', label="Visibility (miles)")
axes_3.set_title("Visibility (miles)")
fig.show()

fig.savefig("Traffic_vol_vs_dew_point_vs_visibility")

fig_1 = plt.plot(group_df['traffic_volume'].groupby(
    [group_df.index.week]).mean())
fig_1.set_title("Traffic Volume")

fig_2 = catplot(x='dew_point', y='traffic_volume',
                order=[np.arange(1, 10)], data=group_df)

fig_3 = catplot(x='visibility_in_miles', y='traffic_volume',
                order=[np.arange(1, 10)], data=group_df)

# Visualize holiday effect on traffic
fig_4 = catplot(x='is_holiday', y='traffic-volume',
                kind='boxen', data=main_data)

# Standardise and normalize data
cols_to_scale = ["air_pollution_index", "clouds_all", "humidity",
                 "rain_p_h", "temperature",
                 "wind_direction", "wind_speed"]

scaler = MinMaxScaler()

air_pollution_index = main_data.air_pollution_index.values.copy()
air_pollution_index = air_pollution_index.reshape(len(air_pollution_index), 1)
air_pollution_index = scaler.fit_transform(air_pollution_index)

clouds_all = main_data.clouds_all.values.copy()
clouds_all = clouds_all.reshape(len(clouds_all), 1)
clouds_all = scaler.fit_transform(clouds_all)

humidity = main_data.humidity.values.copy()
humidity = humidity.reshape(len(humidity), 1)
humidity = scaler.fit_transform(humidity)

rain_p_h = main_data.rain_p_h.values.copy()
rain_p_h = rain_p_h.reshape(len(rain_p_h), 1)
rain_p_h = scaler.fit_transform(rain_p_h)

temperature = main_data.temperature.values.copy()
temperature = temperature.reshape(len(temperature), 1)
temperature = scaler.fit_transform(temperature)

wind_direction = main_data.wind_direction.values.copy()
wind_direction = wind_direction.reshape(len(wind_direction), 1)
wind_direction = scaler.fit_transform(wind_direction)

wind_speed = main_data.wind_speed.values.copy()
wind_speed = wind_speed.reshape(len(wind_speed), 1)
wind_speed = scaler.fit_transform(wind_speed)

main_data = main_data.drop(cols_to_scale, axis=1)

main_data["air_pollution_index"] = air_pollution_index
main_data["clouds_all"] = clouds_all
main_data["humidity"] = humidity
main_data["rain_p_h"] = rain_p_h
main_data["temperature"] = temperature
main_data["wind_direction"] = wind_direction
main_data["wind_speed"] = wind_speed

# Clean up
del [air_pollution_index, clouds_all, humidity,
     rain_p_h, temperature, wind_direction, wind_speed]

fig, axes = plt.subplots(nrows=8,
                         ncols=1,
                         figsize=(10, 15))  # Always length by breadth

for i, var in enumerate(cols_to_plot):
    sns.distplot(main_data[var], ax=axes[i], axlabel=var)
fig.show()

fig.savefig("After normalisation")

# Generate detailed variable profile report

profile = main_data.profile_report(title="Profile Report")
profile.to_file(output_file="profile_report.html")

# Remove unneeded variables
main_data.drop("snow_p_h", axis=1, inplace=True)
main_data.drop("rain_p_h", axis=1, inplace=True)
# Highly correlated with dew_point, r = 1
main_data.drop("visibility_in_miles", axis=1, inplace=True)

# One hot encoding
onehot_holiday = pd.get_dummies(main_data['is_holiday'], prefix="is_holiday")
onehot_weather = pd.get_dummies(
    main_data['weather_description'], prefix="weather_descr")
onehot_w_type = pd.get_dummies(
    main_data['weather_type'], prefix="weather_type")
onehot_dew_point = pd.get_dummies(main_data['dew_point'], prefix="dew_point")

main_data = pd.concat([main_data, onehot_holiday], axis=1)
main_data = pd.concat([main_data, onehot_weather], axis=1)
main_data = pd.concat([main_data, onehot_w_type], axis=1)
main_data = pd.concat([main_data, onehot_dew_point], axis=1)

cat_cols = ["is_holiday", "weather_description", "weather_type", "dew_point"]

main_data.drop(cat_cols, axis=1, inplace=True)

del [onehot_dew_point, onehot_holiday, onehot_w_type, onehot_weather]
# main_data.reset_index(inplace=True)

# Engineer features
main_data['is_weekend'] = 0

for row in range(len(main_data)):
    if main_data.index.day[row] == 5 or main_data.index.day[row] == 6:
        main_data['is_weekend'][row] = 1
test_set["traffic_volume"] = np.nan

"""
Section: 4
Training and testing
"""

main_data['traffic_volume'] = target

# Collinearity check
rank_vis = rank1d(main_data)
rank_vis.show()

rank_vis = rank2d(main_data)
rank_vis.show()

rank_vis = rank1d(main_data, algorithm='shapiro')
rank_vis.show()

# Remove features causing collinearity
cols_to_drop = ["weather_type_0", "weather_type_1", "weather_type_2", "weather_type_3", "weather_type_4",
                "weather_type_5", "weather_type_6", "weather_type_7", "weather_type_8", "weather_type_9",
                "weather_type_10"]

main_data.drop(cols_to_drop, axis=1, inplace=True)

# Linear regression
train = main_data[:27380]
test = main_data[27380:]

x_train = train.drop("traffic_volume", axis=1)
y_train = train['traffic_volume']
x_test = test.drop("traffic_volume", axis=1)
y_test = test["traffic_volume"]

model = LinearRegression()
model.fit(x_train, y_train)
preds = model.predict(x_test)
print("\nMean absolute error: ", metrics.mean_absolute_error(y_test, preds))

model = Ridge()
model.fit(x_train, y_train)
preds = model.predict(x_test)
print("\nMean absolute error: ", metrics.mean_absolute_error(y_test, preds))

model = LassoCV(alphas=np.logspace(-6, 6, 13))
model.fit(x_train, y_train)
preds = model.predict(x_test)
print("\nMean absolute error: ", metrics.mean_absolute_error(y_test, preds))

models = [LinearRegression(),
          LassoCV(alphas=np.logspace(-6, 6, 13)),
          ElasticNetCV(alphas=np.logspace(-6, 6, 13)),
          SGDRegressor(),
          PassiveAggressiveRegressor(),
          Ridge(),
          PassiveAggressiveRegressor(),
          RandomForestRegressor(max_depth=5),
          GradientBoostingRegressor(),
          AdaBoostRegressor(loss='exponential'),
          BaggingRegressor(),
          SVR(),
          NuSVR(),
          XGBRFRegressor(max_depth=5, objective="reg:squarederror"),
          XGBRegressor(max_depth=5, objective="reg:squarederror")]


def show_score(x, y, estimator):
    """
    Returns MAE scores for specified models.
    Also returns r2 scores if applicable

    Arguments:
        x {[array/DataFrame]} -- [Array or matrix of features. Can also be dataframe]
        y {[array]} -- [Target values]
        estimator {[str]} -- [The estimator being used]
    """
    # Instantiate models and predict values
    model = estimator
    model.fit(x, y)
    preds = model.predict(x_test)
    preds = abs(preds.astype(int))
    actuals = y_test

    # Print results
    print(f"{estimator.__class__.__name__}:: r2 score = {round(metrics.r2_score(actuals, preds), 2)} : MAE = {round(metrics.mean_absolute_error(actuals, preds), 2)}")


for model in models:
    show_score(x_train, y_train, model)

figure, axes = plt.subplots(
    nrows=len(models), ncols=2, figsize=(9, 9), sharex=True)

for ind, model in enumerate(models):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    for index, ax in enumerate(axes):
        residuals_plot(model, x_test, preds, hist=False, ax=ax[index])
        prediction_error(model, x_test, preds, ax=ax)

# Do some scoring on XGB estimators
# Validation curve
viz = ValidationCurve(
    XGBRegressor(objective="reg:squarederror"), param_name="max_depth",
    param_range=np.arange(1, 11), cv=5, scoring="r2"
)
viz.fit(x_train, y_train)
viz.show()

# Learning curve
model = XGBRegressor(objective="reg:squarederror")
viz_2 = LearningCurve(
    model, scoring="r2")
viz_2.fit(x_train, y_train)
viz_2.show()

model = RFECV(LassoCV(), cv=5, scoring='r2')
model.fit(x_train, y_train)
model.show()

"""
Section: 5
Time-Series Algorithms
"""
# Fitting ARIMA
# Original Series
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
fig, axes = plt.subplots(3, 1, sharex=True)
plot_acf(main_data.traffic_volume, ax=axes[0])

# 1st Differencing
plot_acf(main_data.traffic_volume.diff(), ax=axes[1])

# 2nd Differencing
plot_acf(main_data.traffic_volume.diff().diff(), ax=axes[2])

plt.show()


# Confirm if d is required
result = adfuller(main_data.traffic_volume)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])


# Show number of p terms required
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(main_data.traffic_volume.diff()
             )
axes[0].set_title("1st Differencing")
axes[1].set(ylim=(0.5))
plot_pacf(main_data.traffic_volume.diff().dropna(), ax=axes[1])

plt.show()

# Start SARIMAX cross validation
pList = [0, 1, 2, 4, 8, 10]
dList = [0, 1, 2, 3, 4]
qList = [0, 1, 2, 3]


def show_bestScore(train_set, test_set):
    """
    Returns best cross-validated
    MAE and (p,d,q) order
    for a ts model.
    """
    start = input("Do you have p, d and q values defined? ")
    if start == "No" or start == "no" or start == "N" or start == "n":
        print("Please define p, d, q values and retry.")
    else:
        print("Finding out...")
        target = [values for values in train_set]
        testVals = [values for values in test_set]
        target = train_set.astype("float32")
        testVals = test_set.astype("float32")
        score = [10000, (0, 0, 0)]
        for p in pList:
            for d in dList:
                for q in qList:
                    order = (p, d, q)
                    model = SARIMAX(target, order=order)
                    fit = model.fit(disp=False)
                    preds = fit.forecast(len(test_set))
                    error = mean_absolute_error(testVals, preds)
                    if score[0] != 0 and error < score[0]:
                        score.pop()
                        score.pop()
                        score.append(error)
                        score.append(order)

        best_score, best_order = score[0], score[1]
        out = print("Best SARIMAX: MAE = %.f :: Order = %s" %
                    (best_score, best_order))
        if not best_score:
            print("Invalid or missing value for MAE. Please retry.")
        elif not best_order:
            print("Invalid or missing order of values. Please retry.")
        else:
            return out  # MAE = 1702 :: Order = (8, 3, 1)

