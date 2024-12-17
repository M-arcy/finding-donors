
# Supervised Learning
## Project: Finding Donors for CharityML

## Project Overview
CharityML is a fictional non-profit organization that aims to optimize outreach to potential donors. By analyzing census data, this machine learning project identifies individuals who are most likely to donate based on income levels, reducing overhead costs and maximizing impact.

This project utilizes supervised learning techniques to build a predictive model that determines whether individuals earn more than $50,000 annually.

#### Objective
The goal of this project is to:

* Build and evaluate multiple supervised learning models to predict potential donors.
* Select the most efficient and accurate model through performance metrics.
* Optimize the final model using grid search for hyperparameter tuning.
* Assess feature importance and determine key predictors for donor identification.


### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Data

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)


### Methodology
**Data Exploration**

* Analyzed 32,000+ records with 13 features.
* Calculated benchmarks using a naive predictor model.

#### Data Preprocessing

Log-transformed skewed continuous features.
Normalized numerical features using MinMaxScaler.
Performed one-hot encoding for categorical data.
Model Selection and Evaluation

Tested three supervised learning models:
Decision Tree Classifier
Random Forest Classifier
AdaBoost Classifier
Evaluated models based on accuracy and F-beta score.
Model Optimization

Optimized the best-performing model (AdaBoost) using GridSearchCV.
Tuned hyperparameters for improved accuracy and F-score.
Feature Importance Analysis

Identified the top 5 predictive features using feature importance.
Tested model performance on a reduced feature set.
Tools and Libraries




### Key Features
Multiple Supervised Learning Models

Decision Trees, Random Forest, and AdaBoost models were implemented and compared.
Feature Engineering

Log transformations handled skewed continuous features.
MinMax scaling normalized numerical data.
One-hot encoding prepared categorical features.
Model Performance Metrics

Accuracy, precision, recall, and F-beta scores were evaluated to select the best model.
Feature Importance

Top 5 most predictive features were identified, simplifying the model without losing much accuracy.
**Results**
|Metric      |	Naive Predictor  |	Unoptimized AdaBoost	| Optimized AdaBoost|
|------------|-------------------|------------------------|-------------------|
|Accuracy	   |    0.2478         |	0.8576	              |    0.8622         |
| F-score	      0.2917           |	0.7246 	              |    0.7348         |

**Top 5 Features:**

capital-gain
age
capital-loss
hours-per-week
education-num
Performance on Reduced Features:
Training on only the top 5 features achieved 83.37% accuracy with an F-score of 0.6789, demonstrating the efficiency of feature reduction.

#### Benefits to CharityML 🎉
By employing the AdaBoost Classifier, CharityML can:

1. Maximize Donations:
Accurately identify high-income individuals most likely to donate, improving outreach efficiency.

2. Reduce Overhead Costs:
Sending donation requests only to relevant individuals saves resources.

3. Data-Driven Decision Making:
Feature importance analysis highlights key attributes for donor prediction, offering valuable insights.

4. Scalable Solution:
The optimized model balances accuracy and computational efficiency, making it practical for large datasets.

#### Conclusion
This project demonstrates the power of supervised learning in solving real-world problems like donor identification for CharityML. The AdaBoost model, through thoughtful preprocessing, optimization, and evaluation, emerges as the most effective solution, providing actionable insights and measurable benefits to maximize donations.

For further improvements, additional techniques like ensemble stacking or feature selection using SHAP values can be explored.

