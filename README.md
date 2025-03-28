# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Janathul firdhous
RegisterNumber:  212224040129
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
DATA HEAD

![Screenshot 2025-03-28 081924](https://github.com/user-attachments/assets/3d0bf59f-eccd-400c-bf02-3d02a2c8bae7)

DATA HEAD1

![Screenshot 2025-03-28 082028](https://github.com/user-attachments/assets/cae769c0-2823-4236-b3b4-0a3d930287e2)

ISNULL().SUM()

![Screenshot 2025-03-28 082131](https://github.com/user-attachments/assets/cc2c3daa-8823-4850-8964-401853472572)


DATA DUPLICATE

![Screenshot 2025-03-28 082239](https://github.com/user-attachments/assets/ef88fb86-28ab-4323-acac-26be350291d9)


PRINT DATA

![Screenshot 2025-03-28 082329](https://github.com/user-attachments/assets/6ab1ad17-8724-46ed-9199-81837777aeb8)


STATUS

![Screenshot 2025-03-28 082432](https://github.com/user-attachments/assets/f45b10bc-942e-4cce-b078-b2da8f866786)


Y_PRED

![Screenshot 2025-03-28 082502](https://github.com/user-attachments/assets/7ee421e3-cf59-4bde-af87-3c619d41eddf)

ACCURACY

![Screenshot 2025-03-28 082549](https://github.com/user-attachments/assets/dec764a3-b779-4ad2-bf90-eabcb986772c)


CONFUSION MATRIX

![Screenshot 2025-03-28 082628](https://github.com/user-attachments/assets/3d884a12-548f-4e61-a254-eea86838840f)


CLASSIFICATION

![Screenshot 2025-03-28 082711](https://github.com/user-attachments/assets/62b61b50-be6d-45ae-8a8e-3413034b022f)


LR PREDICT

![Screenshot 2025-03-28 082745](https://github.com/user-attachments/assets/3982d7af-6f92-4037-aa3c-4178ded37509)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
