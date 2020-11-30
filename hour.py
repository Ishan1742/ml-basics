import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

# print data
bike_df = pd.read_csv('data/hour.csv')
print(f"Data:")
print(bike_df.head())
print()
print(f"Datatypes:")
print(bike_df.dtypes)
print()
print(f"Description:")
print(bike_df.describe())
print()

# check missing values
print("Missing Values:")
print(bike_df.isnull().sum())
print()

bike_df['datetime'] = bike_df[['dteday', 'hr']].apply(lambda x: x['dteday'] + " " + (
    str(x['hr']) if len(str(x['hr'])) == 2 else "0" + str(x['hr'])) + ":00", axis=1)
bike_df['datetime'] = pd.to_datetime(bike_df.datetime, format="%Y-%m-%d %H:%M")

bikers = bike_df.cnt
bikers.index = bike_df.datetime
plt.figure()
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Count based on hour for the first 7 days')
plt.plot(bikers[:24*7])
plt.savefig('hour-results/hour.png')
plt.clf()
print(f"Count based on hour saved: 'hour-results/hour.png'")
print()

bike_df.weathersit = bike_df.weathersit.astype(str)
sns.barplot(x='weathersit', y='cnt', data=bike_df)
plt.title('Count based on weather')
plt.savefig('hour-results/weather.png')
plt.clf()
print(f"Count based on weather saved: 'hour-results/weather.png'")
print()

sns.scatterplot(x='casual', y='registered', data=bike_df)
plt.title('Registered vs Casual')
plt.savefig('hour-results/registered.png')
plt.clf()
print(f"Registered vs Casual saved: 'hour-results/registered.png'")
print()

sns.barplot(x='weekday', y='casual', data=bike_df)
plt.title('Count based on weekday')
plt.savefig('hour-results/weekday.png')
plt.clf()
print(f"Count based on weekday saved: 'hour-results/weekday.png'")
print()

sns.barplot(x='weekday', y='registered', data=bike_df)
plt.title('Count based on weekday registered users')
plt.savefig('hour-results/weekdayregistered.png')
plt.clf()
print(f"Count based on weekday registered users saved: 'hour-results/weekdayregistered.png'")
print()

sns.barplot(x='hr', y='registered', data=bike_df)
sns.barplot(x='hr', y='casual', data=bike_df)
plt.title('Count based on hour')
plt.savefig('hour-results/hourbarplot.png')
plt.clf()
print(f"Count based on hour: 'hour-results/hourbarplot.png'")
print()

plt.figure(figsize=(18, 13))
sns.heatmap(bike_df.corr(), annot=True)
plt.title('Correlation matrix')
plt.savefig('hour-results/correlation.png')
plt.clf()
print(f"Correlation Matrix saved: 'hour-results/correlation.png'")
print()

# categorical variables to dummy variables
X = pd.DataFrame.copy(bike_df)
dcols = ['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']
for col in dcols:
    dummies = pd.get_dummies(X[col], prefix=col)
    X = pd.concat([X, dummies], axis=1)
print(X.head())
print()

for col in dcols:
    del X[col]
print(X.head())
print()
y = X['cnt']
del X['cnt']
del X['casual']
del X['registered']
del X['instant']
del X['dteday']
del X['datetime']
print(X.head())
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=43)
print(X_train.shape)
print(X_test.shape)
print(X_train.head())
print()
print("Fitting data. Please wait...")
print()

# modelling
regressor = RandomForestRegressor(n_estimators=250, max_features='auto')
regressor.fit(X_train, y_train)
r_score = regressor.score(X_train, y_train)
print(f"Accuracy of the model: {r_score}")
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
plt.savefig('hour-results/features.png')
plt.clf()
print(f"Important features saved: 'hour-results/features.png'")
print()

plt.plot([obj for obj in y_test[:150]], color='b', label='Actual')
plt.plot(y_pred[:150], color='r', label='Predicted')
plt.xlabel('Values')
plt.ylabel('Count')
plt.legend()
plt.title('Actual Count vs Predicted Count')
plt.savefig('hour-results/prediction.png')
plt.clf()
print(f"Actual vs Prediction Results saved: 'hour-results/prediction.png'")
print()

with open('hour-results/output.txt', 'w') as file:
    file.write("Predictions vs Actual: \n\n")
    file.write("    Prediction:         Actual:\n")
    i = 0
    for obj in y_test:
        file.write("    {0:15}     {1}\n".format(y_pred[i], obj))
        i += 1
    file.write("\n")
print("Text format for prediction saved: 'hour-results/output.txt")
print()
