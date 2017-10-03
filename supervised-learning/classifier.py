from sklearn import tree

# [weight, texture] -- 1 for smooth, 0 for bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 for Orange, 1 for Apple
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
classifier = clf.fit(features, labels)

# Test prediction
print(classifier.predict([[160, 0]]))
