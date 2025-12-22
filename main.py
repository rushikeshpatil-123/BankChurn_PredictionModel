# Import Data Manipulation Libraries
import numpy as np 
import pandas as pd 

# Import Data Visualization Libraries
import seaborn as sns 
import matplotlib.pyplot as plt 

# Import Data Warning Libraries
import warnings
warnings.filterwarnings(action = 'ignore')

filepath = 'https://raw.githubusercontent.com/rushikeshpatil-123/BankChurn_PredictionModel/refs/heads/main/data/Churn_Modelling.csv'


# Step 1:  Data Ingestion()
def data_ingestion():
   return pd.read_csv(filepath, engine="python", on_bad_lines="skip")

# step 2: Data Exploration 
from collections import OrderedDict
def exploration(df):
      numerical_col = df.select_dtypes(exclude='object').columns
      categorical_col = df.select_dtypes(include='object').columns

      num = []
      cat = []
      info = []

      # Numerical Features ----------
      for i in numerical_col:
         Q1 = df[i].quantile(0.25)
         Q3 = df[i].quantile(0.75)
         IQR = Q3 - Q1
         LW = Q1 - (1.5 * IQR)
         UW = Q3 + (1.5 * IQR)

         Outlier_Count = ((df[i] < LW) | (df[i] > UW)).sum()
         Outlier_Percentage = (Outlier_Count / len(df)) * 100

         numerical_stats = OrderedDict({
               "Feature": i,
               "Count": df[i].count(),
               "Maximum": df[i].max(),
               "Minimum": df[i].min(),
               "Mean": df[i].mean(),
               "Median": df[i].median(),
               "Q1": Q1,
               "Q3": Q3,
               "IQR": IQR,
               "Lower_Whisker": LW,
               "Upper_Whisker": UW,
               "Outlier_Count": Outlier_Count,
               "Outlier_Percentage": Outlier_Percentage
         })

         num.append(numerical_stats)

      numerical_stats_report = pd.DataFrame(num)

      # Categorical Features ----------
      for j in categorical_col:
         categorical_stats = OrderedDict({
               "Feature": j,
               "Count": df[j].count(),
               "Unique_Values": df[j].nunique(),
               "Value_Counts": df[j].value_counts()
         })
         cat.append(categorical_stats)

      categorical_stats_report = pd.DataFrame(cat)

      # Dataset Info ----------
      for k in df.columns:
         info_stats = OrderedDict({
               "Feature": k,
               "Data_Type": df[k].dtype,
               "Null_Count": df[k].isnull().sum(),
               "Null_Percentage": (df[k].isnull().sum() / len(df)) * 100
         })
         info.append(info_stats)

      info_stats_report = pd.DataFrame(info)

      return numerical_stats_report, categorical_stats_report, info_stats_report

# EDA : Crosstab Function Defination
def crosstab(df):
   crosstab1 = pd.crosstab(index = df['Geography'],columns = [df['Exited'],df['Tenure']],margins = True)
   crosstab2 = pd.crosstab(df['Age'],df['Exited'],margins = True)
   crosstab3 = pd.crosstab(df['Tenure'],df['Exited'],margins = True)
   crosstab4 = pd.crosstab(df['EstimatedSalary'],df['Exited'],margins= True)
   crosstab5 = pd.crosstab(df['HasCrCard'],df['Exited'],margins = True)

   return crosstab1,crosstab2,crosstab3,crosstab4,crosstab5


# Preprocessing function defintion
from sklearn.preprocessing import MinMaxScaler,RobustScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocessing(df):

  X = df.drop(columns = ['RowNumber', 'CustomerId', 'Surname','Exited'],axis = 1)
  y = df['Exited']

  X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                   test_size = 0.3,
                                                   random_state = 0)

  le = LabelEncoder()

  for i in X_train.columns:
    if X_train[i].dtype == 'object':
      X_train[i] = le.fit_transform(X_train[i])  # fit_train is always with seen Data
      X_test[i] = le.transform(X_test[i])  # transform is always with Unseen Data

  sc = RobustScaler()
  X_train = sc.fit_transform(X_train) # fit_train is always with seen Data
  X_test = sc.transform(X_test)  # transform is always with Unseen Data

  smote = SMOTE() # Using Over Sampling Technique
  X_train,y_train = smote.fit_resample(X_train,y_train) # Balancing Seen Data

  return X_train,X_test,y_train,y_test


# Step 4 : Model building function
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier

def model_build(X_train,X_test,y_train,y_test):

  models = {
      'LogisticRegression': LogisticRegression(),
      'DecisionTreeClassifier': DecisionTreeClassifier(),
      'RandomForestClassifier': RandomForestClassifier(),
      'GradientBoostingClassifier': GradientBoostingClassifier(),
      'AdaBoostClassifier': AdaBoostClassifier(),
      'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=10)
  }

  model_performance = []

  for model_name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test,y_pred)

    report = OrderedDict({
        'Model_Name': model_name,
        'Accuracy_Score': accuracy_score(y_test,y_pred),
        'Confusion_Matrix': confusion_matrix(y_test,y_pred),
        'Classification_Report': report})

    model_performance.append(report)

  model_performance = pd.DataFrame(model_performance)
  return model_performance

#function calling 
df = data_ingestion()

numerical_stats_report,categorical_stats_report,info_stats_report = exploration(df)

crosstab1,crosstab2,crosstab3,crosstab4,crosstab5 = crosstab(df)

X_train,X_test,y_train,y_test = preprocessing(df)

model_performance = model_build(X_train,X_test,y_train,y_test)

#print(df)
#print(numerical_stats_report)
#print(categorical_stats_report)
#print(info_stats_report)
#print(crosstab1)
#print(X_train)
print(model_performance)

