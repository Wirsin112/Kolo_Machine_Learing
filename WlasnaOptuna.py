from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
class EveningTuna():
    def __init__(self,xtrain,ytrain,xtest,ytest,size=[10,10,10,1],activation="sigmoid",final_activation="sigmoid",learing_rate=[0.4,0.04,0.004,0.0004],beta=1,epochs=10000):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.size = size
        self.activation = activation
        self.final_activation = final_activation
        self.learing_rate = learing_rate
        self.beta = beta
        self.epochs = epochs

    def Eveninglution(self):
        best_stats_jpg = ""
        best_val = 0
        for i in range(self.size[0]):
            for j in range(self.size[0]):
                for k in range(self.size[0]):
                    for l in self.learing_rate():
                        classif = Sequential()
                        classif.add(Dense(i,activation=self.activation,input_dim=(2,)))
                        classif.add(Dense(j, activation=self.activation))
                        classif.add(Dense(k, activation=self.activation))
                        classif.add(Dense(1, activation=self.activation))
                        classif.compile(optimizer=SGD(learning_rate=l),loss="mean_squared_error",metrics=["accuracy"])
                        classif.fit(self.xtrain,self.ytrain,batch_size=1,epochs=self.epochs,callbacks=[EarlyStopping(monitor="loss")])
                        siema = classif.evaluate(self.xtest,self.ytest)
                        if siema > best_val:
                            best_val = siema
                            best_stats_jpg = "Size = ["+str(i)+","+str(j)+","+"k"+",1] | activation = "+str(self.activation)+" | learing rate:"+str(l)
        print(best_stats_jpg)
if __name__ == "__main__":
    pass