# Arvato_customer_segmentation

## Libraries:

* numpy 
* pandas
* matplotlib
* seaborn
* sklearn
* imblearn
* xgboost

## Motivation  

Nowadays, with big data becomes reality, people now focus on how to use the data to realize commercial values. One area which is much more mature is how to picture the potential customer or predict the behavior of the customer, to target the market or customer more precisely.
Bertelsman Arvato Finaincial Solution has provided a challenge: how to target the population and convert them into a mailout company's customers more efficiently.

## Files included:
etl.py: supported files for data cleaning, plot, pca and KMeans.  
Arvato_customer_segment.ipynb: main file


## Summary:
The work I have implemented in this project summarized as below:

* Exploratory the demographics data: general population of Germany, customers of a mail-order company, and market campaign train and test datasets.
* Cleaned the dataset and select features.
* Used PCA and KMeans to cluster the population. 
* Analyzed the higher potential clusters' top 10 features to get a relatively clear picture of the target population
* Applied supervised machine learning with RandomForestClassifier, LogisticsRegression, and XGBClassifier, GridsearchCV to develop a model that can label the positive response probability to a market campaign. 

## Takesaways:
Through the iteration process, I realized that the different estimators might improve the performance a bit, but in order to approach the real limitation, I still have to come back to the first step to process the data more precisely, which is the basis of improving the model performance.

My blogpost is here: [blog](https://towardsdatascience.com/bertelsman-arvato-financial-solution-customer-segmentation-c8528d5ac77a?sk=7af6bc445efccdd52c3acadf38154b46)



```python

```
