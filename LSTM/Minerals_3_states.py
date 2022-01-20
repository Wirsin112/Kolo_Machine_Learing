import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
def switch_to_up_down(column):
    a = []
    for i in range(days):
        a.append(np.NaN)
    for i in range(days,len(df[column])):
        if df[column].values[i-days] < df[column].values[i]*0.95:
            a.append(0)
        elif df[column].values[i]*1.05 > df[column].values[i-days] > df[column].values[i]*0.95  :
            a.append(1)
        else:
            a.append(2)

    return a
if __name__ == "__main__":
    days = 90
    dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'kernel': ['poly', 'rbf', 'linear']
            }
    df = pd.read_csv('materials.csv')
    del df["date"]

    i = 0
    while i < df.shape[1]:
        column_name = df.columns[0]
        df["new"] = 0
        df['new'] = switch_to_up_down(column_name)
        del df[column_name]
        df.rename(columns={"new":column_name}, inplace=True)
        i += 1
    df = df.iloc[90:]

    R = df["molybden-high"]
    print(R)
    del df['molybden-high']
    for i in range(1):
        scaler = MinMaxScaler()
        xtrain, xtest, ytrain, ytest = train_test_split(df.values, R.values, train_size=0.8,shuffle=False)

        ytest = ytest.reshape(-1,1)
        ytrain = ytrain.reshape(-1,1)
        print(ytest)
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
        model = SVC(kernel='linear', C=1)
        model.fit(xtest, ytest)
        # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
        # grid.fit(xtrain, ytrain)
        # print(grid.best_params_)
        b = model.score(ytrain, ytest)
        print(i)
    print(b)
    bobas = model.predict(xtest)
    f = open('file2.pkl', 'wb')
    pickle.dump(bobas, f)