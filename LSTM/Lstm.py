from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df = pd.read_csv("materials.csv")
    print(df.shape)
    dfx = df[["copper-high","molybden-high"]]
    del df["date"]
    xtrain, xtest, ytrain, ytest = train_test_split(dfx.values, df["molybden-high"].values, shuffle=False, train_size=0.8)
    xtrain,xvalid,ytrain,yvalid = train_test_split(xtrain,ytrain,shuffle=False,train_size=0.8)

    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    ytrain = ytrain.reshape(-1, 1)
    yvalid = yvalid.reshape(-1, 1)
    ytest = ytest.reshape(-1, 1)

    xtrain = scalerx.fit_transform(xtrain)
    xvalid = scalerx.transform(xvalid)
    xtest = scalerx.transform(xtest)

    ytrain = scalery.fit_transform(ytrain)
    yvalid = scalery.transform(yvalid)
    ytest = scalery.transform(ytest)

    traingenx = TimeseriesGenerator(xtrain,ytrain,30)
    traingenv = TimeseriesGenerator(xvalid,yvalid,30)
    traingeny = TimeseriesGenerator(xtest,ytest, 30)

    model = Sequential()
    model.add(LSTM(30,input_shape=(30,2),return_sequences=True))
    model.add(LSTM(15))
    model.add(Dense(1,activation="linear"))
    model.compile(Adam(),loss="mse")
    a = EarlyStopping(patience=10)
    modelcheckpoint = ModelCheckpoint("model.h5",save_best_only=True,save_weights_only=True)
    model.fit(traingenx,epochs=1000000,callbacks=[a,modelcheckpoint],validation_data=traingenv)
    a = model.predict(traingeny)
    a = a.T[0]
    ytest = ytest.T[0]
    ytest = ytest[-len(a):]
    print(a,ytest)
    plt.figure()
    plt.plot(a,label="Pred")
    plt.plot(ytest,label="True")
    plt.legend()
    plt.show()
