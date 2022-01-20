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
def count_each(column,df):
    a = {'0':0,'1':0}
    for i in range(len(df[column])):
        if df[column].values[i] == 1:
            a["1"] = a["1"]+1
        elif 0 == df[column].values[i]:
            a["0"] = a["0"]+1
    return a
def switch_to_up_down(column):
    a = []

    for i in range(len(df[column])-days):
        if df[column].values[i+days] < df[column].values[i]:
            a.append(1)
        else:
            a.append(0)
    for i in range(days):
        a.append(0)
    return a
if __name__ == "__main__":
    days = 30
    dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'kernel': ['poly', 'rbf', 'linear']
            }
    df = pd.read_csv('materials.csv')
    print(df.tail(10).to_string())
    del df["date"]

    i = 0
    while i < df.shape[1]:
        column_name = df.columns[0]
        df["new"] = 0
        df['new'] = switch_to_up_down(column_name)
        del df[column_name]
        df.rename(columns={"new":column_name}, inplace=True)
        i += 1
    print(df.head(5).to_string())
    df = df.iloc[:-days]
    # figure = plt.figure(figsize=(12, 8))
    # for x in range(df.shape[1]-1):
    #     print(x)
    #     sns.barplot(x="molybden-high", y=df[df.columns[x]], data=df)
    #     plt.show()

    xnew = df.iloc[:int(len(df)*0.8)]
    alfa = xnew.loc[xnew["molybden-high"] == 0]
    beta = xnew.loc[xnew["molybden-high"] == 1]
    xd = min([alfa.shape[0], beta.shape[0]])
    xnewtop = pd.concat([alfa.iloc[:xd], beta.iloc[:xd]])
    ynewtop = xnewtop["molybden-high"]

    print(count_each("molybden-high", xnewtop))
    del xnewtop["molybden-high"]
    a = []
    R = df["molybden-high"]
    del df["molybden-high"]
    for i in range(1):
        scaler = MinMaxScaler()
        xtrain, xtest, ytrain, ytest = train_test_split(df.values, R.values, train_size=0.8, shuffle=False)
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
        model = SVC(kernel='linear', C=0.1)
        model.fit(xtrain, ytrain)
        # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
        # grid.fit(xtrain, ytrain)
        # print(grid.best_params_)
        b = model.score(xtest, ytest)
        # print(new,yone)
        a.append(b)
        print(i)
    print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))
    bobas = model.predict(xtest)
    f = open('file.pkl','wb')
    pickle.dump(bobas,f)