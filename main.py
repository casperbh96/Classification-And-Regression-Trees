from decision_trees import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load classification data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# initialize tree, fit, predict and score using classification tree
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

# load regression data
# ...

# initialize tree, fit, predict and score using regression tree
# ...