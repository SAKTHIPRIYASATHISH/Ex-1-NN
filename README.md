<H3>ENTER YOUR NAME</H3> S.Sakthi Priya
<H3>ENTER YOUR REGISTER NO.</H3> 212222040140
<H3>EX. NO.1</H3>
<H3>DATE</H3> 24/2/24
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
## IMPORT LIBRARIES:
~~~
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
~~~
## READ THE DATA:
~~~
df = pd.read_csv('Churn_Modelling.csv')
print(df)
~~~
## CHECK DATA:
~~~
df.head()
df.tail()
df.columns
~~~
## CHECK THE MISSING DATA:
~~~
print(df.isnull().sum())
~~~
## CHECK FOR DUPLICATES:
~~~
df.duplicated()
~~~
## ASSIGNING X:
~~~
X = df.iloc[:, :-1].values
print(X)
~~~
## ASSIGNING Y:
~~~
y = df.iloc[:,-1].values
print(y)
~~~
## HANDLING MISSING VALUES:
~~~
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
~~~
## CHECK FOR OUTLIERS:
~~~
df.describe()
~~~
## DROPPING STRING VALUES DATA FROM DATASET: & CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
~~~
df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()
~~~
## NORMALIE THE DATASET USING (MinMax Scaler):
~~~
scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)
~~~
## SPLIT THE DATASET:
~~~
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)
~~~
## TRAINING AND TESTING MODEL:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))
~~~


## OUTPUT:
## DATA CHECKING:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/1c797454-f6a6-4262-b64d-124b8ecef33b)

## MISSING DATA:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/d23571b2-ccdc-458f-9365-a916033e40dd)

## DUPLICATES IDENTIFICATION:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/f44fab68-bf0a-4f9d-9e37-9e39cf8b2101)

## VALUE OF Y:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/60520739-678a-4a91-99b7-6b9c610e308c)

## OUTLIERS:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/f85a4bd7-f48f-46d5-ab5c-4b7eb05b44fa)

## CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![image](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/21910d0e-136f-462c-ba85-7ff5464ce528)

## NORMALIZE THE DATASET:
![Screenshot 2024-02-24 182408](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/68af11b6-4343-430d-986a-c28b4bbecb94)

## SPLIT THE DATASET:
![Screenshot 2024-02-24 182450](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/ea29c965-aa92-40ff-9f4f-e1adc0f7e5f5)

## TRAINING AND TESTING MODEL:
![Screenshot 2024-02-24 182546](https://github.com/SAKTHIPRIYASATHISH/Ex-1-NN/assets/119104282/403152bd-8849-4a39-b5ee-d36f4e5b9b87)













## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


