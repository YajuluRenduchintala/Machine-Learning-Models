#Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Dataset
def dataset():
    try :
        df = pd.read_csv(r'C:\Users\agnih\Desktop\ML\insurance.csv')
    except FileNotFoundError as e :
        print("The following file is not found")
        exit()
    print(df.describe())
    print(df.head())
    df = df.drop_duplicates()
    print(df.isnull().sum())
    data(df)
#Data Correction
def data(df):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    le = LabelEncoder()
    df.iloc[:,1] = le.fit_transform(df.iloc[:,1]) # Converting Sex Column
    df.iloc[:,4] = le.fit_transform(df.iloc[:,4]) #Converting Smoker Column
    from sklearn.compose import ColumnTransformer
    oHe = OneHotEncoder()
    ct=ColumnTransformer(transformers=[("encoder",oHe,['region'])],remainder='passthrough')
    df_transformed = ct.fit_transform(df)
    df_transformed = pd.DataFrame(df_transformed)
    print(df_transformed)
    X=df_transformed.iloc[:,:-1].values #Independent
    y=df_transformed.iloc[:,-1].values  #Dependent
    visualization(df,df_transformed)
    regression(X,y)


#Visualization
def visualization(df, df_transformed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Male vs Female
    s1 = df['sex'].value_counts()
    axes[0].pie(s1, labels=s1.index, autopct='%1.1f%%', shadow=True)
    axes[0].set_title("Male vs Female Percentage")

    # Smoker vs Non-Smoker
    s2 = df['smoker'].value_counts()
    axes[1].pie(s2, labels=s2.index, autopct='%1.1f%%', shadow=True)
    axes[1].set_title("Smoker Vs Non Smoker")

    plt.show()

    # Finding Correlation
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_transformed.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Transformed Features")
    plt.show()
    
def regression(X,y):
    #Linear Regression
    
    from sklearn.model_selection import train_test_split,cross_val_score
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    
    #Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    scores = cross_val_score(rf, X, y, cv=10)
    print('Mean R^2 score:', np.mean(scores))

dataset()
