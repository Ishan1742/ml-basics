import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

# read data from file
attributes = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'species']
iris = pd.read_csv('data/iris.data', names=attributes)
iris.columns = attributes

# print details about the dataset
print("Iris Dataset:")
print(iris)
print()
print()
print("Iris Datatypes:")
print(iris.dtypes)
print()
print()
print("Iris Description:")
print(iris.describe())
print()
print()
print("Iris Species:")
print(iris.groupby('species').size())
print()
print()

# split into training and testing set
train, test = train_test_split(
    iris, test_size=0.3, random_state=43, stratify=iris['species'])
print("Training Set:")
print(train.groupby('species').size())
print()
print()

# histograms
n_bins = 10
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(train['sepal_length'], bins=n_bins)
axs[0, 0].set_title('Sepal Length')
axs[0, 1].hist(train['sepal_width'], bins=n_bins)
axs[0, 1].set_title('Sepal Width')
axs[1, 0].hist(train['petal_length'], bins=n_bins)
axs[1, 0].set_title('Petal Length')
axs[1, 1].hist(train['petal_width'], bins=n_bins)
axs[1, 1].set_title('Petal Width')
fig.tight_layout(pad=1.0)
fig.savefig('iris-results/histogram.png')
print("Histogram Plotted: 'iris-results/histogram.png'")
print()

# pairplot
fig = sns.pairplot(train, hue='species', height=2, palette='colorblind')
fig = fig.fig
fig.savefig('iris-results/pairplot.png')
plt.clf()
print("Pair Plot: 'iris-results/pairplot.png'")
print()

# correlation matrix
corrmat = train.corr()
fig = sns.heatmap(corrmat, annot=True, square=True)
fig = fig.get_figure()
fig.savefig('iris-results/correlation.png')
plt.clf()
print("Correlation Matrix: 'iris-results/correlation.png'")
print()

# create training data
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

# decision tree
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=68)
decision_tree.fit(X_train, y_train)
prediction = decision_tree.predict(X_test)
print(
    f'Accuracy of Decision Tree: {metrics.accuracy_score(prediction, y_test)}')
print()
print(f'Feature Importances: {decision_tree.feature_importances_}')
print()
plt.figure()
fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plot_tree(decision_tree, feature_names=fn, class_names=cn, filled=True)
plt.savefig('iris-results/decisiontree.png')
plt.clf()
print("Decision Tree saved: 'iris-results/decisiontree.png'")
print()

with open('iris-results/output.txt', 'w') as file:
    file.write("Predictions vs Actual: \n\n")
    file.write("    Prediction:         Actual:\n")
    i = 0
    for obj in y_test:
        file.write("    {0:15}     {1}\n".format(prediction[i], obj))
        i += 1
    file.write("\n")

# confusion matrix
disp = metrics.plot_confusion_matrix(
    decision_tree, X_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('Decision Tree Confusion matrix, without normalization')
plt.savefig('iris-results/confusion.png')
plt.clf()
print("Confusion Matrix saved: 'iris-results/confusion.png'")
print()

# prediction plot
plot_colors = "rgb"
n_class = 3
for i, color in zip(cn, plot_colors):
    temp = np.where(y_test == i)
    idx = [elem for elems in temp for elem in elems]
    plt.scatter(X_test.iloc[idx, 2], X_test.iloc[idx, 3], c=color,
                label=i, cmap=plt.cm.RdYlBu, edgecolor='black', s=20)

plt.suptitle("Decision Boundary Shown in 2D with Test Data")
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.legend()
plt.axis("tight")
plt.savefig('iris-results/prediction.png')
plt.clf()
print("Prediction saved: 'iris-results/prediction.png'")
print()
print("Text format for prediction saved: 'iris-results/output.txt")
print()
