# Pipeline-iEEG-Analysis

## Table of content
* [Global Informations](#glob-inf)
* [Aim and scope of the project](#aim)

# Global Informations
This project originate from my Master's thesis, defended in 2021 at the university Lyon II, and supervised by Royce Anders.

# Aim and scope of the project
This project aim to propose a Data Driven alternative to classical method on iEEG analysis, with the help of Machine Learning.
Our PipeLine consist in

* Preprocessing the Data to obtain vectors with each features labeled.
* Reducing the vectors size using univariate feature selection.
* Selecting features based on the fitted models and the shapley value of the features
* Dissect feature contributon to the model using SHAP

This Pipeline allows us to perform trials by trials analysis of iEEG data, and to obtain strong evidence to the link between behavioral and neural data.

