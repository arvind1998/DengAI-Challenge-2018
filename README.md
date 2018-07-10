<h1> DengAI Challenge 2018 </h1>

The files to the challenge can be found [here](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).
There will be four csv files to be downloaded. 
* dengue_features_train.csv
* dengue_labels_train.csv
* dengue_features_test.csv
* submission_format.csv

<h3>Missing Values (NaNs)</h3>
In the dataset, we can see many missing values (NaNs) which have to be replaced. As the data changes with time, I chose the ffill() function to replace it with the last known value.

<h3>Type of Regression</h3>
As the target variable total_cases is always a non-negative integer, this is a count regression problem.
So now, we have two choices for this: 

1. Poisson Regression
1. Negative Binomial Regression

Poisson is used when the mean and variance of data are equal or close to equal. Negative Binomial (Pascal) is used when they are different. 

<h3>Growth of Dengue in the two cities over time</h3>

![](/images/growth_of_dengue.png)

<h3>Correlation</h3>
As this dataset is big, some features might be strongly correlated to the target variable .i.e. total_cases and some might be weakly correlated. So our job is to drop the features which have a less correlation factor. As this dataset consists data of two different cities, we might need to separate them and treat them as two different datasets. 
On plotting the correlations of the features with the target variable we get the following result: 

**Heatmaps: **


![](/images/corr_sj.png)

![](/images/corr_iq.png)

So we can see that few features have negative correlation. So it's wise to remove those features from the dataset and also those which have less (close to 0) correlation. 
