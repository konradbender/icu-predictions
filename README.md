# ICU Predictions based on vital signs

## The data

This notebook is based on the medical record of patients in an Intensive Care Unit. The training data consists
of 230'000 labelled records of patient's vital signs such as their age, temperature or heart rate.

The data is missing some values. For example, for one patient, the entry looks as follows:

Time |  BUN |  Temp | Hgb | HCO3 | BaseExcess | RRate |Phosphate | WBC | 
--- | --- | --- | --- | --- | --- | --- | --- |  --- | 
3 | 12.0  36.0 | 8.7 | 24.0 | -2.0 | 16.0 |   | 6.3 |
4 |    36.0 |   |   | -2.0 | 16.0 |   |   | 
5 |    36.0 |   |   | 0.0 | 18.0 |   |   |  
6 |    37.0 |   |   | 0.0 | 18.0 |   |   |  
7 |      |   |   |   | 18.0 |   |   |   
8 |    37.0 |   |   |   | 16.0 |   |   |   
9 |    37.0 |   |   |   | 18.0 |   |   |   
10 |   | 37.0 |   |   |   | 18.0 |   |   | 
11 | 12.0 |   | 8.5 | 26.0 |   | 12.0 |   4.6 | 4.7 | 
12 | 12.0 | 38.0 | 8.5 | 26.0 | 0.0 | 18.0 |      | 4.7

In order to deal with these missing values, they were imputed with the median on a per-patient level and later 
over the entire dataset.
## Predictions of necessary tests

First, the patients' records are used to train classifiers that will predict if a patient 
will need some test of not. For this, I used a [Gradient Boosted Classifier](https://en.wikipedia.org/wiki/Gradient_boosting)
to determine probabilities that a patient will use the test. The Receiver Operator Curve can be seen below.

[![roc_curve](https://n.ethz.ch/~kbender/download/misc/github_assets/plot.png)](https://n.ethz.ch/~kbender/download/misc/github_assets/plot.png)

## Predictions of vital signs

Second, the vital signs of the patient at the next hour after the recorded data should be predicted. For this, linear regression is a good
estimate.

