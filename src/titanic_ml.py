import numpy as np
import h2o
h2o.init()
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd

if __name__ == "__main__":

    #cleaning the data
    df = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    df_sur = pd.read_csv("data/gender_submission.csv")
    df_test = df_test.join(df_sur.set_index("PassengerId"), on = "PassengerId", rsuffix="_other")
    for i in [df, df_test]:
        i.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket"], axis = 1, inplace = True)
        i.dropna(inplace=True)

    # Putting data into h2o
    train = h2o.H2OFrame(df)
    test = h2o.H2OFrame(df_test)
    factor_col = ["Pclass", "Sex", "SibSp", "Parch"]
    for i in factor_col:
        train[i] = train[i].asfactor()
        test[i] = test[i].asfactor()
    y = "Survived"
    x = train.columns
    del x[0]

    # setting the parameters to try out
    params = {
        'ntrees' : [10,20,30,40],
        'nbins_cats': [2,4,8],
        'learn_rate' : [1,.1],
        'sample_rate' : [.9, 1],
        'col_sample_rate' : [.9,.8],
        'seed' : [42],
        'stopping_rounds' : [10],
        'stopping_tolerance' : [1e-5],
        'stopping_metric' : ['mse'],
        'balance_classes': [True]
        }
    
    # grid search on GBM
    grid = H2OGridSearch(H2OGradientBoostingEstimator, params)
    grid.train(x=x, y=y, training_frame=train)

    # getting the model
    model = grid.get_grid(sort_by='mse', decreasing=True)[0]
    pred = model.predict(test)