from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN:
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i, data in enumerate(self.X_train):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]


# Loads iris dataset
iris = load_iris()

# Uppercase for matrix, lowercase for vector
X = iris.data
y = iris.target

# Split our dataset by 50%/50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# classifier = KNeighborsClassifier()
classifier = ScrappyKNN()
classifier.fit(X_train, y_train)

# Store predictions to variable
predictions = classifier.predict(X_test)

# Print our predictions accuracy compared to our test targets
print(accuracy_score(y_test, predictions))