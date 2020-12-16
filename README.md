# KDD-COVID19

Corona virus disease (COVID-19) is an infectious disease caused by a newly discovered corona virus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. For our group project we will using the COVID-19 in USA dataset that's found on kaggle. We will be performing data preprocessing (handling missing data, feature sampling, dimensionality reduction), data visualization, training and testing the model, and then come up with a final conclusion on the Covid-19 dataset to predict "Deaths Per 100,000 people"  for different counties in each state of the United States of America based on median household income, unemployment rate, poverty rate, white alone %, black alone %, asian alone %, hispanic %.



# Columns in the Datatset :
State - County's corresponding State

FIPS - County's corresponding unique FIPS code for identification and convenience

County - Name of county, city, parish, borough, census area, etc.

Population 2018 - County's total population 2018

Median Household Income 2018 ($) - The median of household income counted in 2018

Unemployment Rate 2018 (%) - Percent of total population that is unemployed 2018

Poverty 2018 (%) - Percent of total population living under the poverty line 2018

Confirmed Cases - Cumulative Confirmed COVID-19 Cases updated 6/02/2020

Confirmed Deaths - Cumulative COVID-19 Deaths updated 6/02/2020

Confirmed Cases Per 100,000 people - The number of confirmed COVID19 cases in USA per 100000 people

Deaths Per 100,000 people - The number of deaths due to COVID19 in USA per 100000 people

Mortality Rate (%) - Mortality rate of each county in the USA in %.

White Alone (%) - The % of white people in each county 

Black Alone (%) - The % of white people in each county 

Native American Alone (%) - The % of native american people in each county 

Asian Alone (%) - The % of asian people in each county 

Hispanic (%) - The % of hispanic people in each county 

Less than a High School Diploma (%) - The % of people having education less than high school diploma

Only a High School Diploma (%) - The % of people with high school diploma

Some College/Associate's Degree (%) - The % of people with college/Associate's degree

Bachelor's Degree or Higher (%) - The % of people with bachelor's degree


# Group members

Rohit Alavala - 800952197

Sai Bharadwaj Reddy - 801166672

Amruta Deshmukh - 801217189

Rishant Dutt - 801104239


# Research Question


While working on this project we are trying to predict the mortality rate for different counties in each state of the United States of America based on median household income, unemployment rate, poverty rate, white alone %, black alone %, asian alone %, hispanic %.

# CRISP-DM Process
We have undergone these steps as a part of CRISP-DM:

## Business Understanding
COVID-19 is an acute infectious respiratory disease caused by infection with the coronavirus subtype SARS-CoV-2, first detected in Wuhan, China, in December 2019. It is currently spreading worldwide and is considered a pandemic disease. In our project we are trying to predict the impact of various factors like median household income, unemployment rate, poverty rate, white alone %, black alone %, asian alone %, hispanic % affects on the mortality rate.

## Data Understanding Phase
In this phase we are adding to the foundation of Business understanding. In this phase we have identified, collected and analysed the data which will help us to achieve goals.
First, we have loaded the the data source(csv).We also identified the data format, number of records and field identities. The columns which are not useful in achieving the project goal are removed and data reduction is achieved.

## Data Preparation Phase
In this phase we found out how clean is the data i.e how many null values are present in the data. We also looked for the duplicate data in the data source. 

## Modeling Phase
For our modeling, we decided to go with 6 models and not rely on only one model. The 6 models that we have used for our project are RandomForestRegressor, ExtraTreesRegressor, RadientBoostingRegressor, HistGradientBoostingRegressor, Ridge & finally ElasticNet. A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. An extra-trees regressor implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Gradient Boosting for regressor builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function. Histogram-based Gradient Boosting Regression Tree. This estimator is much faster than GradientBoostingRegressor for big datasets (n_samples >= 10 000). This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples. The Ridge model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets)). Lastly, the ElasitcNet model is a regularised regression method that linearly combines both penalties i.e. L1 and L2 of the Lasso and Ridge regression methods. It is useful when there are multiple correlated features. The difference between Lass and Elastic-Net lies in the fact that Lasso is likely to pick one of these features at random while elastic-net is likely to pick both at once.We used 'deaths_per_100,000_people' as our target column as per our instructors feedback instead of the 'mortality_rate_'. Making this changed helped the models perform better, espeically Ridge & ElasticNet models. As you can see from the above graph, ExtraTreesRegressor gave us the best accuracy score followed close by RadientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ElasticNet & Ridge. 

## Evaluation Phase
We used the coefficient of determination, R^2, of the predictions to score each prediction. The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

Given the relatively small sample size of our dataset, coming in at 2959 data samples, we believe that the time each model takes to fit the training data is irrelevant. Focussing solely on performance, the aforementioned coefficient of determination seems to be a perfectly valid estimator of a model's performance, all by itself.

Now, amongst the 6 models tested here, the simpler, linear, models do not score particularly well, averaging a score of 0.6 approximately, this tells us that the data maybe too complex and might require more models that are better equiped to deal with that complexity, thus we selected 4 models namely, RandomForestRegressor, ExtraTreesRegressor, RadientBoostingRegressor, and HistGradientBoostingRegressor.

These models outperformed the linear models by a lot, and gave us a very satisfactory score of about 0.96 on average. Amongst these models the ExtraTreesRegressor performed the best with a score close to 0.99, nearly perfect. Hence, in conclusion of this section, we pick the ExtraTreesRegressor as the model of choice to predict, the deaths per 100k people, for the given dataset.


# Technologies/Libraries
python

pandas, jupyter

scikit-learn

numpy

seaborn

# Summary
The World Health Organisation (WHO) has declared the coronavirus disease 2019 (COVID-19) a pandemic. A global coordinated effort is needed to stop the further spread of the virus. A pandemic is defined as “occurring over a wide geographic area and affecting an exceptionally high proportion of the population.” Our dataset state_data has information like median household income, unemployment rate, poverty rate, white alone %, black alone %, asian alone %, hispanic %, etc about all states in the United States of America.
The dataset was impure hence we performed data cleaning and preprocessing before performing any visualization on that. In data cleaning we looked for some missing values as well as for duplicate values. We have also performed data reduction by removing the unnecessary cloumns from the dataset.
In the next step we performed exploratory data analysis on the data. We represented the data in different plots and visualized the data. Going forward we performed different modeling on the data to predict the death rate. The modeling techniques like RandomForestRegressor,ExtraTreesRegressor,RadientBoostingRegressor,HistGradientBoostingRegressor,Ridge,ElasticNet are used.

# Deployment
1. Create a repository on github.
2. Create a jupyter notebook and export it as an html file
3. Add, Commit, & Push changes to GitHub
4. Enable The Project Website

# Discussion
By Fitting the Dataset with different regression models we were able to estimate "the number of deaths per 100000 people", in each county very efficiently. 'Extra Trees Regressor' happens to be the best regression model with a R-squared score of 0.98. 
R-squared gives you the percentage variation in y explained by x-variables. The range is 0 to 1 (i.e. 0% to 100% of the variation in y can be explained by the x-variables).

# Future Work
We have done our project on all the states of United states of America. This can be extended to perform globally. In future, we can consider all the states of other countries of the world as well.


# Resources
https://www.kaggle.com/ady123/us-counties-covid19-dataset

https://towardsdatascience.com/covid-19-outbreak-prediction-using-machine-learning-algorithm-ce5641bd55bf

https://medium.com/@ageitgey/four-basic-data-science-lessons-illustrated-by-covid-19-data-7d94134a5b0e
