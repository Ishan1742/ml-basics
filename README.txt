=========================================
# Artificial Intelligence Assignment 2
# Ishan Ghosh
=========================================
Regression on bike sharing dataset.
Classification on iris dataset.


=========================================
# Installation
=========================================

- Create a virtual environment for python
```bash
$ sudo apt-get install python3-venv
$ cd B170473CS-ISHAN/
$ python3 -m venv env
```

- Activate virtual environment
* make sure you are in the same directory as env/ *
```bash
$ source env/bin/activate
```

- Install required libraries
```bash
$ pip3 install -r requirements.txt
```

- Execute the code
```bash
$ python3 day.py
$ python3 hour.py
$ python3 iris.py
```

- Exit from virtual environment
```bash
$ deactivate
```


=========================================
# Input Data
=========================================
Input data is stored in the "data/". It contains "day.csv", "hour.csv", "iris.data".


=========================================
# Ouput Data
=========================================
There are 3 different types of output data.
- Relevant ouput during execution is printed in the console.
- Graphs created during execution are stored in "[name]-results/". That is "hour-results/". "day-results", "iris-results".
- The predictions or the classfication output files are also stored in the above respective directories with the file name as "output.txt"


=========================================
# 1. Hourly Prediction "hour.csv"
=========================================
Prediction of hourly bike rental count based on the environmental and seasonal settings.
The steps involved in "hour.py" are explained below.

The csv file is read and data extracted to a pandas dataframe and printed to the console.
The data is checked for any missing values.
A datetime entry is created in the data and the date "dteday" and hour "hr" are converted to a single datetime object.
A graph of the count of the rentals and the time for 7 days is plotted in the image "hour-results/hour.png".
A graph of the count based on weather situation is plotted in the image "hour-results/weather.png". We can see that most of the cyclists go out in clear and cloudy weather.
A graph plot between casual and registered riders is plotted. "hour-results/registered.png"
A graph plot between the weekday and count of casual riders is plotted in "hour-results/weekday.png", which shows that casual riders usually rent during weekends.
A graph plot between weekday and registered riders is plotted in "hour-results/weekdayregistered.png".
A graph based on the count og riders both registered and casual is plotted in "hour-results/hourbarplot.png".
The correlation matrix is plotted in "hour-results/correlation.png".

Categorical data is converted to dummy variables. This allows a single regression equation to represent multiple groups.
Various other columns which are not required for regression are removed.
The training data and testing data is split into two different sets.

GridSearchCV() is performed to find the best number of estimators. The code for GridSearchCV is commented out from line "119-129" as it takes a very long time. If it is required to perform again please uncomment those lines to run the code.

A random forest model is built using the tuned hyperparameters and the training set data is fit to it. The accuracy score or R2 score of the model is calculated.
Other parameters such as the Mean Absolute Error and the Root Mean Squared Error are calculated.

The feature importances of the regressor is plotted in the graph "hour-results/features.png"

A graph of predicted vs actual values for 150 values are saved in image "hour-results/prediction.png"
The values of the predicted vs actual are also saved in a file "hour-results/output.txt".

=========================================
# 1. Daily Prediction "day.csv"
=========================================
Prediction of daily bike rental count based on the environmental and seasonal settings.
The steps involved in "day.py" are almost similar to "hour.py" except the changes in some of the graphs that are plotted. The images are saved in "day-results/".

The data is read from a file.
Various graphs are plotted in "day-results/".
Outliers are imputed by taking the 75 25 percentile and their mean.
Normal probability plot and correltation matrix is plotted in "day-results/normal.png" and "day-results/correlation.png" respectively.

The testing and training data is split and unneeded variables are removed.
GridSearchCV() is performed. Please uncomment lines "169-179" in "day.py" to execute grid search.

The random forest regressor is created using the tuned parameters and this model is fit to the training data.

The accuracy, mean absolute error and root mean squared error is calculated.

The feature importance is plotted in a graph "day-results/features.png".

The predicted vs actual values for 150 values are saved in a graph "day-results/prediction.png"
The values of the predicted vs actual are also saved in a file "day-results/output.txt".

=========================================
# 1. Iris Classification "iris.data"
=========================================
Classify a flower into any of the three classes based on the attributes given.
The steps involved in "iris.py" is explained below.

The data is extracted from "iris.data". Appropriate values about the data are printed to the console.
The data is split into training set and testing set.

The different type of data values are plotted in the histogram in "iris-results/histogram.png".
All values are pairwise plotted in "iris-results/pairplot.png"
The correlation matrix is plotted in "iris-results/correlation.png"

The species column is removed from the training and testing data.
A Decision Tree Classifier is used to fit the training data. The accuracy importance of the features are printed to the console.

The decision tree is saved in "iris-results/decisiontree.png".

The confusion matrix is saved in "iris-results/confusion.png".

The actual vs predicted values are saved in "iris-results/output.txt", and the prediction graph is plotted to "iris-results/prediction.png"
