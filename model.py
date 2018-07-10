# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Importing the dataset
train_features=pd.read_csv('dengue_features_train.csv')
train_label=pd.read_csv('dengue_labels_train.csv')
test_features=pd.read_csv('dengue_features_test.csv')

# Appending the target variable to train_features
train_features['total_cases']=train_label.total_cases

# Visualizing the growth of Dengue over time
sns.set(style="ticks", palette="colorblind")
plot = sns.FacetGrid(train_features, hue='city', aspect=4) 
plot.map(sns.pointplot,'weekofyear','total_cases')
plot.add_legend()
plot.fig.suptitle("Growth of Dengue over time")

# Separating the training data of the two cities
train_features_sj = train_features[train_features.city == 'sj'].copy()
train_features_iq = train_features[train_features.city == 'iq'].copy()

# Replacing NaNs with the previously occured value
train_features_sj=train_features_sj.ffill()
train_features_iq=train_features_iq.ffill()

# Separating the test data of the two cities
test_features_sj = test_features[test_features.city == 'sj'].copy()
test_features_iq = test_features[test_features.city == 'iq'].copy()

# Replacing the NaNs with the previosuly occured value
test_features_sj=test_features_sj.ffill()
test_features_iq=test_features_iq.ffill()

# Finding the correlation matrix
sj_corr=train_features_sj.corr()
iq_corr=train_features_iq.corr()

# Plotting the bar graphs of the correlations of features to total_cases
sj_corr.total_cases.drop('total_cases').sort_values(ascending=True).plot.barh(title='SJ: Correlations of features with total_cases')
iq_corr.total_cases.drop('total_cases').sort_values(ascending=True).plot.barh(title='IQ: Correlations of features with total_cases')

# Removing the columns with negative/less correlation from SJ training set
train_features_sj.drop('reanalysis_tdtr_k', axis=1, inplace=True)
train_features_sj.drop('year', axis=1, inplace=True)
train_features_sj.drop('ndvi_ne', axis=1, inplace=True)
train_features_sj.drop('ndvi_se', axis=1, inplace=True)
train_features_sj.drop('station_diur_temp_rng_c', axis=1, inplace=True)
train_features_sj.drop('ndvi_sw', axis=1, inplace=True)
train_features_sj.drop('station_precip_mm', axis=1, inplace=True)
train_features_sj.drop('reanalysis_sat_precip_amt_mm', axis=1, inplace=True)
train_features_sj.drop('precipitation_amt_mm', axis=1, inplace=True)

# Removing the columns with negative/less correlation from IQ training set
train_features_iq.drop('reanalysis_tdtr_k', axis=1, inplace=True)
train_features_iq.drop('ndvi_ne', axis=1, inplace=True)
train_features_iq.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)
train_features_iq.drop('ndvi_se', axis=1, inplace=True)
train_features_iq.drop('station_diur_temp_rng_c', axis=1, inplace=True)
train_features_iq.drop('weekofyear', axis=1, inplace=True)
train_features_iq.drop('ndvi_nw', axis=1, inplace=True)
train_features_iq.drop('station_precip_mm', axis=1, inplace=True)
train_features_iq.drop('ndvi_sw', axis=1, inplace=True)

# Removing the columns with negative/less correlation from SJ test set
test_features_sj.drop('reanalysis_tdtr_k', axis=1, inplace=True)
test_features_sj.drop('year', axis=1, inplace=True)
test_features_sj.drop('ndvi_ne', axis=1, inplace=True)
test_features_sj.drop('ndvi_se', axis=1, inplace=True)
test_features_sj.drop('station_diur_temp_rng_c', axis=1, inplace=True)
test_features_sj.drop('ndvi_sw', axis=1, inplace=True)
test_features_sj.drop('station_precip_mm', axis=1, inplace=True)
test_features_sj.drop('reanalysis_sat_precip_amt_mm', axis=1, inplace=True)
test_features_sj.drop('precipitation_amt_mm', axis=1, inplace=True)

# Removing the columns with negative/less correlation from IQ test set
test_features_iq.drop('reanalysis_tdtr_k', axis=1, inplace=True)
test_features_iq.drop('ndvi_ne', axis=1, inplace=True)
test_features_iq.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)
test_features_iq.drop('ndvi_se', axis=1, inplace=True)
test_features_iq.drop('station_diur_temp_rng_c', axis=1, inplace=True)
test_features_iq.drop('weekofyear', axis=1, inplace=True)
test_features_iq.drop('ndvi_nw', axis=1, inplace=True)
test_features_iq.drop('station_precip_mm', axis=1, inplace=True)
test_features_iq.drop('ndvi_sw', axis=1, inplace=True)

# Getting the shape of the matrix to find the split
train_features_sj.shape # (936, 16)
train_features_iq.shape # (520, 17)

# Splitting the training set of SJ into training set and validation set
train_features_sj_train= train_features_sj.head(800)
train_features_sj_valid= train_features_sj.tail(train_features_sj.shape[0] - 800)
# Splitting the training set of IQ into training set and validation set
train_features_iq_train= train_features_iq.head(400)
train_features_iq_valid = train_features_iq.tail(train_features_iq.shape[0] - 400)

def find_model(training_set, validation_set, city):
                    
    if city=='sj':
        formula="total_cases~1 + " \
                 "weekofyear + " \
                 "reanalysis_specific_humidity_g_per_kg + " \
                 "reanalysis_dew_point_temp_k + " \
                 "station_avg_temp_c + "\
                 "reanalysis_max_air_temp_k + " \
                 "station_max_temp_c + " \
                 "reanalysis_min_air_temp_k + " \
                 "reanalysis_air_temp_k"
                 
    elif city=='iq':
        formula="total_cases~1 + " \
                "reanalysis_specific_humidity_g_per_kg + " \
                "reanalysis_dew_point_temp_k + " \
                "reanalysis_min_air_temp_k + " \
                "station_min_temp_c " 
              
        
    
    grid_search=10**np.arange(-10, -2, dtype=np.float64)              
    best_score = 50
        
    for alpha in grid_search:
        model=smf.glm(formula=formula, data=training_set, family=sm.families.NegativeBinomial(alpha=alpha))
        results=model.fit()
        prediction=results.predict(validation_set).astype(int)
        score=eval_measures.meanabs(prediction, validation_set.total_cases)

        if score<best_score:
            best_alpha=alpha
            best_score=score
            
    print('Final Score = ',best_score)
            
    final_dataset=pd.concat([training_set, validation_set])
    model=smf.glm(formula=formula, data=final_dataset, family=sm.families.NegativeBinomial(alpha=best_alpha))
    final_model=model.fit()
    return final_model

# Finding the final and best model for both the cities
sj_final_model=find_model(train_features_sj_train, train_features_sj_valid, train_features_sj.city[0])
iq_final_model=find_model(train_features_iq_train, train_features_iq_valid, train_features_iq.city[936])

# Overviewing the model
sj_final_model.summary()
iq_final_model.summary()

# Plotting the Actual vs Prediction graph for both the cities
_ , axes = plt.subplots(nrows=2, ncols=1)
train_features_sj['fitted'] = sj_final_model.fittedvalues
train_features_sj.fitted.plot(ax=axes[0], label="Prediction")
train_features_sj.total_cases.plot(ax=axes[0], label="Actual")
train_features_iq['fitted'] = iq_final_model.fittedvalues
train_features_iq.fitted.plot(ax=axes[1], label="Prediction")
train_features_iq.total_cases.plot(ax=axes[1], label="Actual")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

# Making the final predictions
sj_final_predictions = sj_final_model.predict(test_features_sj).astype(int)
iq_final_predictions = iq_final_model.predict(test_features_iq).astype(int)

# Reading the submission_format.csv and removing the total_cases column
submission = pd.read_csv("submission_format.csv")
submission.drop('total_cases', axis=1, inplace=True)

# Adding the predicted values to the dataframe
submission['total_cases'] = pd.concat([sj_final_predictions, iq_final_predictions])

# Saving the dataframe as a csv file
submission.to_csv("Final_Submission.csv")
