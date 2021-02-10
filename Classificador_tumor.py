from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas

# importando os dados
data = load_breast_cancer()

#  Organizando as informacoes
label_names = data['target_names']
labels = data['target']
features_names = data['feature_names']
features = data['data']

#  print(pandas.DataFrame(features, columns=features_names))

#  Segmentando os dados
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#############################################
# testando o algoritmo Naive Bayes (NB)
gnb = GaussianNB()
model = gnb.fit(x_train, y_train)

preds = gnb.predict(x_test)
# print(preds)
print(accuracy_score(y_test, preds))

#############################################
# testando o algoritmo Arvore de decisao
cart = DecisionTreeClassifier()
model1 = cart.fit(x_train, y_train)

preds1 = cart.predict(x_test)
# print(preds1)
print(accuracy_score(y_test, preds1))

##############################################
# Testando o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=1)
model2 = knn.fit(x_train, y_train)

preds2 = knn.predict(x_test)
print(accuracy_score(y_test, preds2))
