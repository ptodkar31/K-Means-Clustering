# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:58:10 2024

@author: Priyanka
"""
"""
Perform clustering analysis on the telecom dataset. The data is a
 mixture of both categorical and numerical data. It consists of the 
 number of customers who churn. Derive insights and get possible information
 on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset.
Business Problem :-
Q.What is the business objective?
Customer segmentation is critical for auto insurance companies to gain competitive advantage by mining useful customer related information.
 While some efforts have been made for customer segmentation to support auto insurance decision making, 
 their customer segmentation results tend to be affected by the characteristics of the algorithm used and lack multiple validation from multiple algorithms.
Q.Are there any constraints?
there is mixed data,categorical and numerical data

"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
# Now import file from data set and create a dataframe
autoi=pd.read_csv("C:\Data Set\AutoInsurance.csv")
#EDA
autoi.info
autoi.dtypes
autoi.describe()
#The average customer lifetime value is 8004 and min is 1898 and max is 83325

# As follwing columns are  going to contribute hence drop it
autoi1=autoi.drop(["Customer","State","Education","Sales Channel","Effective To Date"],axis=1)

plt.hist(data = autoi1, x = 'Customer Lifetime Value');
#This is apparently not a normal distribution.
# And with one peak indicate customer lifetime value of 100000 is higher
plt.hist(data = autoi1, x = 'Income');
#This is apparently not a normal distribution.lower income customers are more
plt.hist(data = autoi1, x = 'Monthly Premium Auto');
# lower premium customers are more


# There are several columns having ctegorical data,so create dummies for these
  #for all these columns create dummy variables
Response_dummies=pd.DataFrame(pd.get_dummies(autoi1['Response']))
Coverage_dummies=pd.DataFrame(pd.get_dummies(autoi1['Coverage']))
Employment_Status_dummies=pd.DataFrame(pd.get_dummies(autoi1['EmploymentStatus']))
Gender_dummies=pd.DataFrame(pd.get_dummies(autoi1['Gender']))
LocationCode_dummies=pd.DataFrame(pd.get_dummies(autoi1['Location Code']))
Marital_Status_dummies=pd.DataFrame(pd.get_dummies(autoi1['Marital Status']))
Policy_Type_dummies=pd.DataFrame(pd.get_dummies(autoi1['Policy Type']))
Policy_dummies=pd.DataFrame(pd.get_dummies(autoi1['Policy']))
Renew_Offer_Type_dummies=pd.DataFrame(pd.get_dummies(autoi1['Renew Offer Type']))
Vehicle_Class_dummies=pd.DataFrame(pd.get_dummies(autoi1['Vehicle Class']))
Vehicle_Size_dummies=pd.DataFrame(pd.get_dummies(autoi1['Vehicle Size']))


## now let us concatenate these dummy values to dataframe
autoi_new=pd.concat([autoi1,Response_dummies,Coverage_dummies,Employment_Status_dummies,Gender_dummies,LocationCode_dummies,Marital_Status_dummies,Policy_Type_dummies,Policy_dummies,Renew_Offer_Type_dummies,Vehicle_Class_dummies,Vehicle_Size_dummies],axis=1)

autoi_new=autoi_new.drop(["Response","Coverage","EmploymentStatus","Gender","Location Code","Marital Status","Policy Type","Policy","Renew Offer Type","Vehicle Class","Vehicle Size"],axis=1)
# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to crime datframe for all the rows 
#and column from 1 until end
    
df_norm=norm_func(autoi_new.iloc[:,:])
# you can check the df_norm dataframe which is scaled between values from 0 to1
# you can apply describe function to new data frame
df_norm.describe()

# Now apply this normalization function to airlines datframe for all 
#the rows and column from 1 until end

df_norm=norm_func(autoi_new.iloc[:,:])
TWSS=[]
k=list(range(2,26))
# The values generated by TWSS are 24 and two get x and y values 24 by 24 ,
#range has been changed 2:26
#again restart the kernel and execute once
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS

plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_clusters");plt.ylabel("Total_within_SS")
# from the plot it is clear that the TWSS is reducing from k=2 to 3 and 3 to 4 
#than any other change in values of k,hence k=3 is selected
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
autoi_new['clust']=mb
autoi_new.head()
autoi_new=autoi_new.iloc[:,[51,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]

autoi_new.iloc[:,:].groupby(autoi_new.clust).mean()

autoi_new.to_csv("kmeans_autoi_new.csv",encoding="utf-8")
import os
os.getcwd()