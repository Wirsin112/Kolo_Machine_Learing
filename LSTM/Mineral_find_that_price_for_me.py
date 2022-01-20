import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pickle


def switch_to_up_down(column):
    a = []

    for i in range(len(df[column])-days):
        if df[column][i] != 0:
           a.append(min([2, df[column][i+1]/df[column][i]]))
        else:
            a.append(0)
    for i in range(days):
        a.append(0)
    return a
def spin_me_right_round(x):
    return round(x,2)

def add_all_pre():
    i = 0
    honolulu = []
    Kubus_puchatek = 0
    for x in range(df.shape[0]):
        Kubus_puchatek = df["gallium-high"][i] + df["cobalt-high"][i] +df["chrom-high"][i] + df["brent_oil-high"][i] + df["arsenic-high"][i] + df["germanium-high"][i] + df["selen-high"][i] + df["copper-high"][i] + df["zinc-high"][i] + df["manganese-high"][i]
        honolulu.append(Kubus_puchatek)
    return honolulu


if __name__ == "__main__":
    days = 30
    dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'kernel': ['poly', 'rbf', 'linear']
            }
    df = pd.read_csv('materials.csv')
    print(df.tail(10).to_string())
    del df["date"]
    f = open('file.pkl','rb')
    bobas = pickle.load(f)
    f = open('file2.pkl', 'rb')
    bobasd = pickle.load(f)
    i = 90
    while i < df.shape[1]-1:
        column_name = df.columns[0]
        df["new"] = 0
        df['new'] = switch_to_up_down(column_name)
        del df[column_name]
        df.rename(columns={"new":column_name}, inplace=True)
        # df[column_name] = df[column_name].map(spin_me_right_round)
        i += 1
    print(df.tail(20).to_string())
    #

    # for x in range(df.shape[1]-1):
    #     print(x)
    #     sns.barplot(x="molybden-high", y=df[df.columns[x]], data=df)
    #     plt.show()
    # print(df["molybden-high"].unique())
    df["new-moly"] = df["molybden-high"].shift(-days)
    print(df.to_string())
    df = df.iloc[:-days]
    a = []

    R = df["new-moly"]
    del df["new-moly"]
    for i in range(1):
        scaler = MinMaxScaler()
        xtrain, xtest, ytrain, ytest = train_test_split(df.values, R.values, train_size=0.80, shuffle=False)
        # xtrain = np.reshape(xtrain,(1,len(xtrain)))
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
        model = LinearRegression(positive=True)
        model.fit(xtrain, ytrain)
        b = model.score(xtest, ytest)
        a.append(b)
        print(i)
    print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))
    my_prediction = model.predict(xtest)
    figure = plt.figure(figsize=(12, 8))
    plt.plot(scaler.fit_transform(np.reshape(my_prediction,(len(my_prediction),-1))),label="Siano")
    plt.plot(scaler.fit_transform(np.reshape(ytest,(len(ytest),-1))),label="Tak powinno byc")
    plt.plot(bobasd, label="down_up3 states")
    plt.legend()
    plt.show()