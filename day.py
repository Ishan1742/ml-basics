import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
# warnings.filterwarnings('ignore')

# print and rename data
bike_df = pd.read_csv('data/day.csv')
print(f"Shape of the dataset: {bike_df.shape}")
print()
print("Data types: ")
print(bike_df.dtypes)
print()
print("Data: ")
print(bike_df.head(5))
print()
bike_df.rename(columns={'instant': 'bike_id', 'dteday': 'datetime', 'yr': 'year', 'mnth': 'month',
                        'weathersit': 'weather_condition', 'hum': 'humidity', 'cnt': 'total_count'}, inplace=True)
print("Renamed data: ")
print(bike_df.head(5))
print()

# type cast data
bike_df['datetime'] = pd.to_datetime(bike_df.datetime)
bike_df['season'] = bike_df.season.astype('category')
bike_df['year'] = bike_df.year.astype('category')
bike_df['month'] = bike_df.month.astype('category')
bike_df['holiday'] = bike_df.holiday.astype('category')
bike_df['weekday'] = bike_df.weekday.astype('category')
bike_df['workingday'] = bike_df.workingday.astype('category')
bike_df['weather_condition'] = bike_df.weather_condition.astype('category')
print("Description: ")
print(bike_df.describe())
print()

# check missing values
print("Missing Values:")
print(bike_df.isnull().sum())
print()

# Histograms
fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(x='month', y='total_count', data=bike_df[[
            'month', 'total_count', 'season']], hue='season', ax=ax)
ax.set_title('Season monthly distribution')
plt.savefig('day-results/season.png')
plt.clf()
print("Seasonwise distribution: 'day-results/season.png'")
print()

fig, ax1 = plt.subplots(figsize=(15, 8))
sns.barplot(x='month', y='total_count', data=bike_df[[
            'month', 'total_count', 'weekday']], hue='weekday', ax=ax1)
ax1.set_title('Weekday monthly distribution')
plt.savefig('day-results/weekday.png')
plt.clf()
print("Weekday distribution: 'day-results/weekday.png'")
print()

# Violin Plot
fig, ax = plt.subplots(figsize=(15, 8))
sns.violinplot(x='year', y='total_count',
               data=bike_df[['year', 'total_count']])
ax.set_title('Yearly distribution')
plt.savefig('day-results/year.png')
plt.clf()
print("Yearly distribution: 'day-results/year.png'")
print()

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(data=bike_df, x='holiday', y='total_count', hue='season')
ax.set_title('Holiday distribution')
plt.savefig('day-results/holiday.png')
plt.clf()
print("Holiday distribution: 'day-results/holiday.png'")
print()

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(data=bike_df, x='workingday', y='total_count', hue='season')
ax.set_title('Workingday wise distribution of counts')
plt.savefig('day-results/workday.png')
plt.clf()
print("Workingday distribution: 'day-results/workday.png'")
print()

# outliers
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=bike_df[['temp', 'windspeed', 'humidity']])
ax.set_title('Temp_windspeed_humidity_outiers')
plt.savefig('day-results/outliers.png')
plt.clf()
print("Outliers: 'day-results/outliers.png'")
print()


# Replace and impute outliers
# create dataframe for outliers
wind_hum = pd.DataFrame(bike_df, columns=['windspeed', 'humidity'])
# Cnames for outliers
cnames = ['windspeed', 'humidity']

for i in cnames:
    # Divide data into 75%quantile and 25%quantile.
    q75, q25 = np.percentile(wind_hum.loc[:, i], [75, 25])
    iqr = q75-q25  # Inter quantile range
    min = q25-(iqr*1.5)  # inner fence
    max = q75+(iqr*1.5)  # outer fence
    wind_hum.loc[wind_hum.loc[:, i] < min, :i] = np.nan  # Replace with NA
    wind_hum.loc[wind_hum.loc[:, i] > max, :i] = np.nan  # Replace with NA
# Imputating the outliers by mean Imputation
wind_hum['windspeed'] = wind_hum['windspeed'].fillna(
    wind_hum['windspeed'].mean())
wind_hum['humidity'] = wind_hum['humidity'].fillna(wind_hum['humidity'].mean())
# Replacing the imputated windspeed
bike_df['windspeed'] = bike_df['windspeed'].replace(wind_hum['windspeed'])
# Replacing the imputated humidity
bike_df['humidity'] = bike_df['humidity'].replace(wind_hum['humidity'])
print("Imputed data: ")
print(bike_df.head(5))
print()

# Normal plot
fig = plt.figure(figsize=(15, 8))
stats.probplot(bike_df.total_count.tolist(), dist='norm', plot=plt)
plt.savefig('day-results/normal.png')
plt.clf()
print("Normal Plot: 'day-results/normal.png'")
print()

# Correlation Matrix
# Create the correlation matrix
correMtr = bike_df.corr()
fig = sns.heatmap(correMtr, annot=True, square=True)
fig = fig.get_figure()
fig.savefig('day-results/correlation.png')
plt.clf()
print("Correlation Matrix: 'day-results/correlation.png'")
print()

# Modelling the dataset
X_train, X_test, y_train, y_test = train_test_split(
    bike_df.iloc[:, 0:-3], bike_df.iloc[:, -1], test_size=0.3, random_state=43)
X_train.reset_index(inplace=True)
y_train = y_train.reset_index()
X_test.reset_index(inplace=True)
y_test = y_test.reset_index()

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
print(f"Training data: \n{y_train.head()}")
print(f"Testing data: \n{y_test.head()}")
