# Algerian Forest Fire Analysis

## Overview

This project analyzes the risk of forest fires in Algeria using meteorological data and fire weather indices collected from June to September 2012 across two forested regions: Bejaia and Sidi Bel-Abbas. Motivated by Algeria's susceptibility to forest fires due to its arid climate and landscape, the research repurposes binary fire occurrence data into continuous risk probabilities.

## Data and Methodology

The dataset comprises 243 observations capturing discrete meteorological features (Temperature, Relative Humidity, Wind Speed, Rain) and continuous fire weather indices (FFMC, DMC, DC, ISI, BUI, FWI). A continuous response variable for regression analysis was derived using a k-nearest neighbors (kNN) approach to estimate fire risk probabilities from local neighborhood observations.

Three regression techniques were evaluated:
- Multiple Linear Regression
- Ridge Regression
- Generalized Additive Models (GAM)

## Key Findings
- Fire weather indices outperform meteorological features in predicting fire risk.
- Ridge Regression with fire weather indices achieved robust predictive performance, maintaining lower residual trends and homoskedasticity compared to other methods.
- Generalized Additive Models (GAM) showed high predictive power but displayed notable residual clustering, suggesting unexplained variability.
- Meteorological features alone proved inadequate in effectively explaining forest fire occurrences.

## Conclusions and Recommendations

Ridge regression based on fire weather indices is most effective for modeling forest fire risk. Future studies should investigate individual feature importance, interaction effects, and potential time-series patterns to enhance predictive accuracy.

This project demonstrates the potential of regression modeling to improve forest fire prediction, contributing to more effective prevention strategies.


