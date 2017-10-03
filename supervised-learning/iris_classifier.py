import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

# Loads built-in iris dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set)
iris = load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# Text IO using buffer (use variable as default output instead of a file)
dot_data = StringIO()

# Exporting decision tree to PDF
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True, impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')
