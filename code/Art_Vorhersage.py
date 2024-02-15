import argparse
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description="Analyze iris data")
parser.add_argument('data', help="Input data (CSV) to process")
parser.add_argument('output_figure', help="Output figure path")
parser.add_argument('output_report', help="Output report path")
args = parser.parse_args()

df = pd.read_csv(args.data)
attributes = ["sepal_length", "sepal_width", "petal_length","petal_width", "class"]
df.columns = attributes

plot = sns.pairplot(df, hue='class', palette='muted')
plot.savefig(args.output_figure)

array = df.values
X = array[:,0:4]
Y = array[:,4]
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(    
		X, Y,    
		test_size=test_size,    
		random_state=seed)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

report = classification_report(Y_test, predictions, output_dict=True)
df_report = pd.DataFrame(report).transpose().to_csv(args.output_report)
