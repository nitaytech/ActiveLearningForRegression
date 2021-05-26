# ActiveLearningForRegression

<p align="center">
<img src ="images/al_cycle.jpg">
</p>

## Prerequisites:
1. [Anaconda 3](https://www.anaconda.com/download/)
2. [modAL](https://modal-python.readthedocs.io/en/latest/)
3. [xgboost](https://github.com/dmlc/xgboost)

## Getting Started

1. Pip install [modAL](https://pypi.org/project/modAL/) package: (`pip install modAL`)
2. Pip install [xgboost](https://pypi.org/project/xgboost/) package: (`pip install xgboost`)
3. Clone this [repository](https://github.com/nitaytech/ActiveLearningForRegression): `git clone https://github.com/nitaytech/ActiveLearningForRegression.git`
4. Open the `MACROS.py` file and update the variables in this file according to your reserach needs. The file contains a description of each variable.
5. Run the `main.py` file: `python3 main.py`
6. The results will appear in the [results](https://github.com/nitaytech/ActiveLearningForRegression/results) folder.
7. Use the [notebook](https://github.com/nitaytech/ActiveLearningForRegression/results) to analyze the results.

## The Data
We have uploaded a data file (`data/milk_sessions.csv`) which contains real features of milking sessions of cows. Note that the results of the paper cannot be reproduced by this data file, since we have ommited additional features which we could not disclose for busniess reasons.

### Data Schema (columns):
* 'Date'
* 'CowID'
* 'DailyYield_KG':
* 'DailyFat_P'
* 'DailyProtein_P'
* 'DailyConductivity'
* 'DailyActivity'
* 'CurrentRP'
* 'CurrentMET'
* 'CurrentKET'
* 'CurrentMF'
* 'CurrentPRO'
* 'CurrentLDA'
* 'CurrentMAST'
* 'CurrentEdma'
* 'CurrentLAME'
* 'Disease'
* 'DIM'
* 'DIM_<50', 'DIM_50-175', 'DIM_>=175' - an indicator (0 or 1)
* 'LactationNumber'
* 'Fertility_Num'
* 'Age'
* 'Still'
* 'Milk'
* 'MilkTemperature'
* 'Fat'
* 'Protein'
* 'Lactose'
* 'LogScc'
* 'Cf'
* 'Blood'
* 'Casein'
* 'Mufa'
* 'Pufa'
* 'Sfa'
* 'Ufa'
* 'Pa'
* 'Sa'
* 'Oa'
* 'Component7'

## The Paper
Paper is under review.
[Technical report](https://github.com/nitaytech/ActiveLearningForRegression/blob/main/Tech-Report.pdf)

<p align="center">
<img src ="images/algorithm.jpg">
</p>


**BibTeX**:
T.B.A
