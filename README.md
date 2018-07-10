<h1> DengAI Challenge 2018 </h1>

The files to the challenge can be found [here](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).
There will be four csv files to be downloaded. 
* dengue_features_train.csv
* dengue_labels_train.csv
* dengue_features_test.csv
* submission_format.csv

<h3>Missing Values (NaNs)</h3>
In the dataset, we can see many missing values (NaNs) which have to be replaced. As the data changes with time, I chose the ffill()function to replace it with the last known value.

<h3>Type of Regression</h3>
As the target variable total_cases is always a non-negative integer, this is a count regression problem.
So now, we have two choices for this: 

1. Poisson Regression
1. Negative Binomial Regression

Poisson is used when the mean and variance of data are equal or close to equal. Negative Binomial (Pascal) is used when they are different. 

<h3>Growth of Dengue in the two cities over time</h3>

![](/images/growth_of_dengue.png)

<h3>Correlations</h3>
As this dataset is big, some features might be strongly correlated to the target variable .i.e. total_cases and some might be weakly correlated. So our job is to drop the features which have a less correlation factor. As this dataset consists data of two different cities, we might need to separate them and treat them as two different datasets. 

```
train_features_sj.total_cases.mean()
> 34.18
train_features_sj.total_cases.var()
> 2640.04
train_features_iq.total_cases.mean()
> 7.56
train_features_iq.total_cases.var()
> 115.89
```
As mean and variance of total_cases is absolutely different for both the cities, we will use the **Negative Binomial Regression**.


On plotting the correlations of the features with the target variable we get the following result: 

![](/images/corr_sj_heatmap.png)

![](/images/corr_iq_heatmap.png)

![](/images/corr_sj.png)

![](/images/corr_iq.png)

So we can see that few features have negative correlation. So it's wise to remove those features from the dataset and also those which have less (close to 0) correlation. 

<h3>Train-Test Split</h3>
The training set for both cities is split into a training set and a validation set. 

 ```
 train_features_sj.shape
 > (936, 16)
 train_features_iq.shape
 > (520, 17)
 ``` 
 So for SJ a good split will be 800-136 and for IQ it will be 400-120.
 
 <h3>Final Model</h3>
 We make the model using Negative Binomial of Generalized Linear Model (GLM) class. Refer the code for further understanding. 
 After making the predictions on the validation set, we plot it.
 
 ![](/images/actual_vs_prediction.png)
 
 <h3>Requirements</h3>
 
 * Numpy
 * Pandas
 * Matplotlib
 * Seaborn
 * Statsmodels
 * Scikit-learn
 
 <h5>To run the model: python model.py</h5>
 

<h3>Results</h3>

1. The meanabs() error for SJ: 20.58
1. The meanabs() error for IQ: 10.26
