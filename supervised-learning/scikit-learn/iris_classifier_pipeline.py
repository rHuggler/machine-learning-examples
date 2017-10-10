from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loads iris dataset
iris = load_iris()

# Uppercase for matrix, lowercase for vector
X = iris.data
y = iris.target

# Split our dataset by 50%/50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Store predictions to variable
predictions = classifier.predict(X_test)

# Print our predictions accuracy compared to our test targets
print(accuracy_score(y_test, predictions))