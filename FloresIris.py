#  importando as bibliotecas

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Carregando os dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

## Analisando o conjunto de dados
# print(f'Dimensao: {dataset.shape}')
# print(f'Amostra: {dataset.head(20)}')
# print(f'Estatistica: {dataset.describe()}')

## Visualizando os dados univariado
# dataset.plot(kind='box', subplots= True, layout= (2,2), sharex= False, sharey= False)
# dataset.hist()
# plt.show()

## Visualizando os dados multivariado
# scatter_matrix(dataset)
# plt.show()
# Obs. quanto mais proximo os pontos mais correlacao ha entre as variaveis

## Dividindo os dados para o treinamento e para a validacao
array = dataset.values
x = array[:, 0:4]  # todas as linhas das 4 primeiras colunas
y = array[:, 4]
validation_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size
                                                                                , random_state=seed)
## Validacao cruzada com 10 partes, treinar 9 e testar 1

## Construindo modelos
# LR: Regrassao linear
# LDA: Analise linear discriminante
# KNN: K-vizinhos mais proximos
# CART: Arvore de classificacao e regessao
# NB: Gaussian Naive Bayes
# SVM: Support Vector Machines
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

seed = 7
scoring = 'accuracy'
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)  # validacao cruzada
    # avalia os modelos estatisticamente
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # print(msg)

## comparando os resultados graficamente
# fig = plt.figure()
# fig.suptitle('Algoritmo Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

## Analisando o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)


## Analisando o algoritmo SVM
# SVM = SVC(gamma='auto')
# SVM.fit(x_train, y_train)
# predictions = SVM.predict(x_validation)

print(y_validation)
print(predictions)
# print(accuracy_score(y_validation, predictions))
# print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

