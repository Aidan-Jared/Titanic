import numpy as np
import h2o
h2o.init()
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
import math

if __name__ == "__main__":

    #cleaning the data
    df = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    df_sur = pd.read_csv("data/gender_submission.csv")
    df_test = df_test.join(df_sur.set_index("PassengerId"), on = "PassengerId", rsuffix="_other")
    for i in [df, df_test]:
        i.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket", "SibSp"], axis = 1, inplace = True)
        i.dropna(inplace=True)

    # Putting data into h2o
    train = h2o.H2OFrame(df)
    train, valid = train.split_frame([.8], seed=62)
    test = h2o.H2OFrame(df_test)
    factor_col = ["Pclass", "Sex", "Parch", "Survived"]
    for i in factor_col:
        train[i] = train[i].asfactor()
        test[i] = test[i].asfactor()
    y = "Survived"
    x = train.columns
    del x[0]

    # setting the parameters to try out
    params = { "ntrees": [10000],
                'max_depth' : list(range(1,30,1)),
                'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'min_rows': [2**x for x in range(0,int(math.log(train.nrow,2)-1)+1)],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(4,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}
    
    # grid search on GBM
    grid = H2OGridSearch(H2OGradientBoostingEstimator, params)
    grid.train(x=x, y=y, training_frame=train, validation_frame=valid, max_runtime_secs=3600)

    # getting the model
    best_model = grid.get_grid(sort_by='auc', decreasing=True)[0]
    perf = best_model.model_performance(test)
    pred = best_model.predict(test)
    print(perf.auc())
    print(pred.head())
    print("done")