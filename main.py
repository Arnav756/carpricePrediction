import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("car data.csv")
#print(df.columns)
#print(df.isnull().sum())

final_dataset= df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
#print(final_dataset.columns)

#print(final_dataset.Year.dtype)
final_dataset['Years_used']= 2020-(final_dataset['Year'])
final_dataset.drop(['Year'],axis=1,inplace=True)
#print(final_dataset.head().to_string())
final_dataset = pd.get_dummies(final_dataset,drop_first=True)
#print(final_dataset.head().to_string())

#sns.pairplot(final_dataset)
corrmat=final_dataset.corr()
top_corr_features=corrmat.index
#plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()

input=final_dataset.drop(['Selling_Price'],axis=1)
target=final_dataset['Selling_Price']
#print(input)

X_train,X_test,Y_train,Y_test=train_test_split(input,target,test_size=0.3)

model=RandomForestClassifier(n_estimators=60)
model.fit(X_train,Y_train.astype('int'))
print(model.score(X_test,Y_test.astype('int')))

file=open('random_forest.pkl','wb')
pickle.dump(model,file)

