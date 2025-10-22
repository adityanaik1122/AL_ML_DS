from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd

X,Y = load_iris(return_X_y=True)

mod = KNeighborsRegressor().fit(X,Y)
pipe = Pipeline([ ("Scale", StandardScaler()) , ("Model", KNeighborsRegressor(n_neighbors = 1)) ])
#pipe.get_params()
pipe.fit(X,Y)
pred = pipe.predict(X)

GridSearchCV(estimator=pipe, param_grid = {'model__n_neighbors' : [1,2,3,4,5,6,7,8,9,10]}, cv = 3)
mod.fit(X,Y)
pd.Dataframe(mod.cv_results_)
plt.scatter(pred,Y)
plt.show()
