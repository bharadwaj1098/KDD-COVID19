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
This work is under progress.

## Evaluation Phase
This work is under progress.


# Technologies/Libraries
python

pandas, jupyter

scikit-learn

numpy

seaborn

# Resources
https://www.kaggle.com/ady123/us-counties-covid19-dataset

https://towardsdatascience.com/covid-19-outbreak-prediction-using-machine-learning-algorithm-ce5641bd55bf

https://medium.com/@ageitgey/four-basic-data-science-lessons-illustrated-by-covid-19-data-7d94134a5b0e
