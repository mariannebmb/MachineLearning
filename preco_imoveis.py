#  importando as bibliotecas

import pandas
from sklearn.model_selection import train_test_split


#  Introduce algorithms.
from sklearn.linear_model import RidgeCV, LinearRegression
#  Integrate algorithms.
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Carregando os dados da biblioteca sklearn
from sklearn.datasets import load_boston

boston = load_boston()
values = boston.data
target = boston.target
names = boston.feature_names

data = pandas.DataFrame(values, columns=names)

#  Segmentando os dados
x_train, x_test, y_train, y_test = train_test_split(values, target, test_size=0.3, random_state=42)

##############################
scores = []
models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('Ridge', RidgeCV(alphas=(0.001,0.1,1),cv=3)))
models.append(('RandomForrest', RandomForestRegressor(n_estimators=10)))
models.append(("GBDT",GradientBoostingRegressor(n_estimators=30)))
models.append(('tree', DecisionTreeRegressor()))

for nome, model in models:
    model_fitted = model.fit(x_train, y_train)
    y_pred = model_fitted.predict(x_test)
    score = r2_score(y_test, y_pred)
    scores.append(score)

    msg = "%s: %f (%f)" % (nome, score.mean(), score.std())
    print(msg)