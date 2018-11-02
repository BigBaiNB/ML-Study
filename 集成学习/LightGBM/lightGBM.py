from lightgbm.sklearn import LGBMClassifier
import pandas as pd
df = pd.read_csv("datas/covtype.data",sep=",",header=None)

## 构建X和Y
X = df.iloc[:,0:-1]
Y = df.iloc[:,-1]

## 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state = 0)

model = LGBMClassifier(max_depth=40,n_estimators=100)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)