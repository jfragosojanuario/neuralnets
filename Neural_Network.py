# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

import os
os.chdir("C:\\Users\\m20170985\\Desktop")


#preprocessing
dataset = pd.read_csv("Churn_Modelling.csv")

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder1 = LabelEncoder()

x[:,1] = labelenconder1.fit_transform(x[:,1])
labelencoder2 = LabelEncoder()
x[:,2]=labelencoder2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1] )
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]


#Splitting dataset into training and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #método que utiliza os dados do set de treino, calcula o maximo e mínimo e faz a normalizaçºao
x_test = sc.transform(x_test) #aqui não posso usar o fit no teste para não viciar o resultado; neste caso estaria a utilizar parametros do teste para treinar o modelo

#NOTA: não temos que normalizar o variável dependente pois é uma prática errada

#criar rede neuronal
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#adicionar a primeira layer
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu", input_dim=11)) #11 variáveis de entrada; uma boa prática de iniciar uma rede contar neurónios como metade do numero de variáveis do dataset , neste caso (11+1)/2=6
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1, kernel_initializer="uniform",activation="sigmoid")) #last layer

#At this point the weights in the network are random hence we need to train the network

#Train the Network

classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"]) #with inbalanced dataset it is preferable to use weighted acuracy
classifier.fit(x_train,y_train,epochs=100) #definir os parâmetros da data de input e o numero de iterações no treino

y_pred = classifier.predict(x_test) 
y_pred=(y_pred>0.5) #criar um treshold; se for superior a 0,5 vale como 1, se for inferior vale como 0

#confusion matrix to assess performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)





