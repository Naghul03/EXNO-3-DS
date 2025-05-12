## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![image](https://github.com/user-attachments/assets/a12c26da-ee74-4b3d-ad43-e7906c0e2a1f)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/b1faca61-cd70-4fc5-b234-00dd481b0763)
```
df['Gh1']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/fa270eac-f960-4668-a32e-8a4d065eb485)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/c614b73a-aaa5-477f-8e72-1afdb6a959a9)
```
from sklearn.preprocessing import OneHotEncoder
ohn=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohn.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/5eb0ceb7-3de6-414a-a386-bd069cc29bf7)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/0763f5e6-0a10-4031-b60d-928bb7d33e4d)
```
pip install --upgrade category_encoders
```

![image](https://github.com/user-attachments/assets/159553ad-ac33-4b6e-b1ce-de6e84a516a3)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```

![image](https://github.com/user-attachments/assets/00636f03-57ad-4427-93a0-c7b9be71ac7d)
```
be=BinaryEncoder()
nd=be.fit_transform(df[['Ord_2']])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/21ae6c26-563d-4f75-8fbc-e590ed8eba6d)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
cc
```
![image](https://github.com/user-attachments/assets/74a788d7-f2c9-48f8-ad69-a7232e95dbd2)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/43e23172-3188-4313-b637-10966ace6d10)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/76a23176-03b2-498c-8608-8053b0b3ee3f)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f46063d2-4af1-406f-9137-c2c08d71528b)
```
np.reciprocal(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b1774462-9b9d-4d57-b964-55f7c6f05b09)
```
np.reciprocal(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/1d3bb20b-7a1d-42b2-8f3f-89d2741169db)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7c8d3388-118b-44e6-bee5-050194f6b65f)
```
df["Highly positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/a8761cde-5490-4fe0-9604-8bf767a1fee7)
```
df['Moderate Negative Skew_yeojohnson'], parameters = stats.yeojohnson(df['Moderate Negative Skew'])
from sklearn.preprocessing import QuantileTransformer
Qt = QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"] = Qt.fit_transform(df[["Moderate Negative Skew"]])
df

```
![image](https://github.com/user-attachments/assets/77f81e53-b660-49aa-ad41-fb2e5fca3489)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/39957220-62ab-4a5a-be23-8ae68640e5ac)
```
sm.qqplot(df["Moderate Negative Skew_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/ee78d891-5410-4aea-8150-cd53e7236fd1)
```
df["Highly Negative Skew_1"] = Qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"], line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/d4ef225c-de47-4348-8e44-b8331f77e7be)
```
sm.qqplot(df["Highly Negative Skew_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/27febd3c-0958-43d7-a833-a72a9dd3b2be)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/dc2ae9ef-863d-4bee-aaca-29ebc83daf24)

# RESULT:
Thus,the feature encoding and transformation process and save the data to the file was
performed successfully.


       
