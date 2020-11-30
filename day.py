import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

bike_df = pd.read_csv('data/day.csv')
print(f"Shape of the dataset: {bike_df.shape}")
print()
print("Data types: ")
print(bike_df.dtypes)
print()
print("Data: ")
print(bike_df.head(5))
print()

print("Description: ")
print(bike_df.describe())
print()

bike_df['dteday'] = pd.to_datetime(bike_df.dteday)
bike_df['season'] = bike_df.season.astype('category')
bike_df['yr'] = bike_df.yr.astype('category')
bike_df['mnth'] = bike_df.mnth.astype('category')
bike_df['holiday'] = bike_df.holiday.astype('category')
bike_df['weekday'] = bike_df.weekday.astype('category')
bike_df['workingday'] = bike_df.workingday.astype('category')
bike_df['weathersit'] = bike_df.weathersit.astype('category')

# check missing values
print("Missing Values:")
print(bike_df.isnull().sum())
print()

# Histograms
plt.figure(figsize=(15, 8))
sns.barplot(x='mnth', y='cnt', data=bike_df[[
            'mnth', 'cnt', 'season']], hue='season')
plt.title('Season monthly distribution')
plt.savefig('day-results/season.png')
plt.clf()
print("Seasonwise distribution: 'day-results/season.png'")
print()

plt.figure(figsize=(15, 8))
sns.barplot(x='mnth', y='cnt', data=bike_df[[
            'mnth', 'cnt', 'weekday']], hue='weekday')
plt.title('Weekday monthly distribution')
plt.savefig('day-results/weekday.png')
plt.clf()
print("Weekday distribution: 'day-results/weekday.png'")
print()

# Violin Plot
plt.figure(figsize=(15, 8))
sns.violinplot(x='yr', y='cnt',
               data=bike_df[['yr', 'cnt']])
plt.title('Yearly distribution')
plt.savefig('day-results/year.png')
plt.clf()
print("Yearly distribution: 'day-results/year.png'")
print()

plt.figure(figsize=(15, 8))
sns.barplot(data=bike_df, x='holiday', y='cnt', hue='season')
plt.title('Holiday distribution')
plt.savefig('day-results/holiday.png')
plt.clf()
print("Holiday distribution: 'day-results/holiday.png'")
print()

plt.figure(figsize=(15, 8))
sns.barplot(data=bike_df, x='workingday', y='cnt', hue='season')
plt.title('Workingday wise distribution of counts')
plt.savefig('day-results/workday.png')
plt.clf()
print("Workingday distribution: 'day-results/workday.png'")
print()

# outliers
plt.figure(figsize=(15, 8))
sns.boxplot(data=bike_df[['temp', 'windspeed', 'hum']])
plt.title('Temp_windspeed_humidity_outiers')
plt.savefig('day-results/outliers.png')
plt.clf()
print("Outliers: 'day-results/outliers.png'")
print()


# Replace and impute outliers
wind_hum = pd.DataFrame(bike_df, columns=['windspeed', 'hum'])
cnames = ['windspeed', 'hum']
for i in cnames:
    q75, q25 = np.percentile(wind_hum.loc[:, i], [75, 25])
    iqr = q75 - q25
    min = q25 - (iqr * 1.5)
    max = q75 + (iqr * 1.5)
    wind_hum.loc[wind_hum.loc[:, i] < min, :i] = np.nan
    wind_hum.loc[wind_hum.loc[:, i] > max, :i] = np.nan
wind_hum['windspeed'] = wind_hum['windspeed'].fillna(
    wind_hum['windspeed'].mean())
wind_hum['hum'] = wind_hum['hum'].fillna(wind_hum['hum'].mean())
bike_df['windspeed'] = bike_df['windspeed'].replace(wind_hum['windspeed'])
bike_df['hum'] = bike_df['hum'].replace(wind_hum['hum'])
print("Imputed data: ")
print(bike_df.head(5))
print()

# Normal plot
plt.figure(figsize=(15, 8))
stats.probplot(bike_df.cnt.tolist(), dist='norm', plot=plt)
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
print()
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
print()
print(f"Training data: \n{y_train.head()}")
print()
print(f"Testing data: \n{y_test.head()}")
print()

train_attributes = X_train[['season', 'mnth', 'yr', 'weekday', 'holiday',
                            'workingday', 'weathersit', 'hum', 'temp', 'windspeed']]
test_attributes = X_test[['season', 'mnth', 'yr', 'weekday', 'holiday',
                          'workingday', 'hum', 'temp', 'windspeed', 'weathersit']]
cat_attributes = ['season', 'holiday',
                  'workingday', 'weathersit', 'yr']
num_attributes = ['temp', 'windspeed', 'hum', 'mnth', 'weekday']

train_encoded_attributes = pd.get_dummies(
    train_attributes, columns=cat_attributes)
print('Shape of training data: ', train_encoded_attributes.shape)
print()
print(train_encoded_attributes.head())
print()

X_train = train_encoded_attributes
y_train = y_train.cnt.values

test_encoded_attributes = pd.get_dummies(
    test_attributes, columns=cat_attributes)
print('Shape test data: ', test_encoded_attributes.shape)
print()
print(test_encoded_attributes.head())
print()

X_test = test_encoded_attributes
y_test = y_test.cnt.values

regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(X_train, y_train)
r_score = regressor.score(X_train, y_train)
print("Accuracy of the model: ", r_score)
print()

r2_scores = cross_val_score(regressor, X_train, y_train, cv=3)
print('R-squared scores :', np.average(r2_scores))
print()

y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print()

rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
print()

feature_importance = regressor.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12, 10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('day-results/features.png')
plt.clf()
print(f"Important features saved: 'day-results/features.png'")
print()

plt.plot([obj for obj in y_test[:150]], color='b', label='Actual')
plt.plot(y_pred[:150], color='r', label='Predicted')
plt.xlabel('Values')
plt.ylabel('Count')
plt.legend()
plt.title('Actual Count vs Predicted Count')
plt.savefig('day-results/prediction.png')
plt.clf()
print(f"Actual vs Prediction Results saved: 'day-results/prediction.png'")
print()

with open('day-results/output.txt', 'w') as file:
    file.write("Predictions vs Actual: \n\n")
    file.write("    Prediction:         Actual:\n")
    i = 0
    for obj in y_test:
        file.write("    {0:15}     {1}\n".format(y_pred[i], obj))
        i += 1
    file.write("\n")
print("Text format for prediction saved: 'day-results/output.txt")
print()
