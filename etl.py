#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#pd.options.mode.chained_assignment = None


# In[3]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


def get_to_know_data(df):
    print("shape: {}".format(df.shape))
    print('')
    print("info: {}".format(df.info()))
    print('')
    print("Thers is {} duplicated records".format(sum(df.duplicated())))


# In[ ]:


def plot1(df):
    '''
    This function is to plot the null value percentage in column wise
    input: dataframe
    
    '''
    df_null=(df.isnull().sum()/df.shape[0]).sort_values(ascending=False)
    df_null=pd.DataFrame(df_null,columns=['percentage'])
    print("null value percentage description: {}".format(df_null.describe()))
    # when customers_null is dataframe, use data and x
    plt.figure(figsize=(15,5))
    bins = np.arange(0, df_null.values.max(), 0.005)
    plt.hist(data = df_null, x = 'percentage', bins = bins)
    plt.title('Null value percentage in column wise')
    plt.xlabel('Null value percentage in column wise')
    plt.ylabel('Column count')
    plt.xticks(np.arange(0, df_null.values.max()+0.05, 0.05));


# In[1]:


def data_clean(val2,df):
    #1. check the most missing value columns
    #as there are two columns, use zip
    # The target is: if a is in azdias columns, for all val1 in b, replace the vala as nan
    # The most intelligent part is for all

    for a,b in zip(val2.feature,val2.new):
         if a in df.columns:
                #print(a)
                for val1 in b:
                    df[a]=df[a].replace(val1,np.nan)
    
    # now check the missing values in columns for azdias,
    # The target is that if the missing values is higher than a certain percentage, the column can be reduced
    
    df_null_col=df.isnull().sum().sort_values(ascending=False)
    df_null_col_per=df_null_col/df.shape[0]
    df_null_list=[keys for keys,values in df_null_col_per.items() if values>0.9]
        
    # del the columns from df_azd_null_list as below
    for i in df_null_list:
        del df[i]
    
    #2. now clean the object data
    # further clean the other value which hasn't been found previously into nan
    df.loc[df['CAMEO_DEU_2015']=='XX', 'CAMEO_DEU_2015']=np.nan
    # Replace the nan with the most frequent value, use mode()
    df.loc[df['CAMEO_DEU_2015'].isnull(), 'CAMEO_DEU_2015']=df['CAMEO_DEU_2015'].mode()[0]
    
    df.loc[df['CAMEO_DEUG_2015']=='X', 'CAMEO_DEUG_2015']=np.nan
    # set nan as the most frequent value
    df.loc[df['CAMEO_DEUG_2015'].isnull(), 'CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].mode()[0]
    # unify the type as int64
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].astype('int64')
    
    df.loc[df['CAMEO_INTL_2015']=='XX', 'CAMEO_INTL_2015']=np.nan
    df.loc[df['CAMEO_INTL_2015'].isnull(), 'CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].mode()[0]
    # unify the type as int64
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].astype('int64')
    
    df.loc[df['D19_LETZTER_KAUF_BRANCHE'].isnull(), 'D19_LETZTER_KAUF_BRANCHE']=df['D19_LETZTER_KAUF_BRANCHE'].mode()[0]
    
    df.loc[df['EINGEFUEGT_AM'].isnull(), 'EINGEFUEGT_AM']=df['EINGEFUEGT_AM'].mode()[0]
    # only care about the inserted year
    df['EINGEFUEGT_AM']=pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    
    df.loc[df['OST_WEST_KZ'].isnull(), 'OST_WEST_KZ']=df['OST_WEST_KZ'].mode()[0]
    df['OST_WEST_KZ'].replace(to_replace=['W','O'],value=[0,1],inplace=True)
    
    #After checking the value attributes, I think CMEO-DEU-2015 is detail info for CMEO-DEUG-2015, which can be represent by it
    #The D19_LETZTER_KAUF_BRANCHE no need to encoding, as it is just summarized of the other info
    #LNR is the identification, can be dropped
    df_index=df['LNR']
    del df['CAMEO_DEU_2015'], df['D19_LETZTER_KAUF_BRANCHE'],df['LNR']
    
    # In my ETL file, I also delete the raw records in row with nan percentage >=0.5, that can be as a practis, here I don't touch it
    
    # fill all of the other nan as column's mean()
    fill_mode = lambda col: col.fillna(col.mode()[0])

    df=df.apply(fill_mode, axis=0)
    
    '''
    drop row with most missing value
    df.ind=[]
    df.loc[(df_azdias_pca.isnull().sum(axis=1).sort_values(ascending=False))/(df.shape[1])>=0.5,'ind']=1
    df.loc[(df_azdias_pca.isnull().sum(axis=1).sort_values(ascending=False))/(df.shape[1])<0.5,'ind']=0
    df['ind']=df_azdias_pca['ind'].astype('int64')
    df.drop(df[df['ind']==1].index)
    '''
    
    
    return df, df_index
    # save cleaned azdias data    
    #df.to_csv('2_cleaned_data.csv',index=False)


# In[ ]:


def do_pca(n_components, data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT: n_components - int - the number of principal components to create
           data - the data you would like to transform

    OUTPUT: 
    pca: pca object after instatiant
    X_pca: new matrix after the instatiant being fit and transformed
    
    X: is not input or output, but better to mention that it is the scaled (standardized) data from the original
    '''
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# In[ ]:


def print_weights(n):
    '''
    n: number of principal component
    '''
    components = pd.DataFrame(np.round(pca.components_[n - 1: n], 4), columns = azdias.keys())
    components.index = ['Weights']
    components = components.sort_values(by = 'Weights', axis = 1, ascending=False)
    components = components.T
    print(components)
    return components


# In[ ]:


def scree_plot_1(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    
    '''
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
    '''
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[ ]:


def scree_plot_2(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    
    
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
    
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[ ]:


# now fit the decomposited data to kmeans
# it takes time
# use elbow method to find the best k
def model_kmeans(X_pca):
    get_ipython().run_line_magic('time', '')
    SSE = []  # 存放每次结果的误差平方和
    for k in range(2, 11):
        #stimator = KMeans(n_clusters=k)  # 构造聚类器
        #estimator.fit(np.array(mdl[['Age', 'Gender', 'Degree']]))
        #SSE.append(estimator.inertia_)
        k_means = KMeans(init = "k-means++", n_clusters = k)
        k_means.fit(X_pca)
        SSE.append(k_means.inertia_)
    X = range(2, 11)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()   
    return k_means


# In[ ]:


def plot_comparision(df):
    #plt.figure(figsize=figsize)
    plt.bar(df.cluster, df.percentage)
    plt.xlabel('Cluster label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('1', fontsize=18)
    #plt.gca().invert_yaxis();

