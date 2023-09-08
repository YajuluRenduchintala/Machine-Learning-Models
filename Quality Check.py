import pandas as pd 
import matplotlib.pyplot as plt 
data_set = pd.read_csv(r'C:\Users\agnih\Desktop\ML\tvmarketing.csv')
print(data_set.columns)

#Checking for null values Quality Check-1
print(data_set.isnull().sum())

#Checking for duplicate values Quality Check-2
print(data_set.duplicated().sum())

#Checking for Outliers in the Columns Quality Check-3
plt.boxplot(data_set['Sales'])
plt.show()

