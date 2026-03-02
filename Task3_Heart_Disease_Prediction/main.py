import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df=pd.read_csv("heart_disease_uci.csv")
# print(df.head())
# print(df.columns)
# print(df.sample(5))
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# print(df['num'].value_counts())

# Moving to step 2
df=df.drop(columns=['id','ca','thal','slope'])

df["num"]=df["num"].apply(lambda x:0 if x==0 else 1)
sns.scatterplot(x='trestbps',y='age',hue='num',data=df)
plt.show()
# print(df.head())
# print(df['num'].value_counts())
# print(df.info())
cat_cols=['sex','dataset','cp','fbs','restecg','exang']
num_cols=['age','trestbps','chol','thalch','oldpeak']

for col in num_cols:
    df[col].fillna(df[col].median(),inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)
print(df.isnull().sum())

# now lets scale it

preprocessor=ColumnTransformer(
    transformers=[
    ('num',StandardScaler(),num_cols),
    ('cat',OneHotEncoder(drop='first'),cat_cols)
])
X=df.drop(columns=['num'])
y=df['num']
X_transformed=preprocessor.fit_transform(X)
print(X_transformed.shape)

X_train,X_test,y_train,y_test=train_test_split(X_transformed,y,
test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print("training completed")
y_pred=model.predict(X_test)

# print(y_test.iloc[22])
# print(y_pred[22])

# Lets go for accuracy now
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

# Confusion matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# Classification Report

print(classification_report(y_test,y_pred))