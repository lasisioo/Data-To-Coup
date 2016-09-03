# Data To Coup

## Requirements: Python 2.7, Anaconda with following python libraries installed

1. numpy
2. pandas
3. sklearn
4. matplotlib
5. patsy


## Models used:
- Random Forest
- AdaBoost
- Logistic Regression

<b>Problem Statement:</b> Considering social and political instabilities that have occurred in recent times, I want predict the likelihood of a coup d'etat within a country, given specific economic, political and social factors.

- Economic factors: income per capita, growth rate
- Social factors: accounts of ethnic violence, religious, linguistic, and ethnic fractionalization, fatalities from 
  violence, urbanization
- Political factors: polity score, length of stay for incumbent head of state


## Data Dictionary:

| Variable | Description | Data Type | Variable Type |
| --- | --- | :---: | --- |
| country | Country Name | Object | Unique |
| coup | 1 = Coup(successful and attempted), 0 = No coup | Integer | Binary |
| gdppercap | GDP per capita | Float | Continuous |
| gdpgrowth | GDP growth rate | Float | Continuous |
| urbanpop | Percentage of urban population | Float | Continuous |
| gendeathmag |Scaled annual number of deaths (range 0-5.0) | Float | Categorical Ordinal |
| revmagfight |Scaled annual number of fatalities related to revolutionary fighting (range 0-4) | Integer | Categorical Ordinal |
| ethmagfatal |Scaled annual number of fatalities related to ethnic fighting (range 0-4)|Integer | Categorical Ordinal |
| revmagfatal |Scaled number of rebel combatants or activists in revolutionary war (range 0-4) | Integer | Categorical Ordinal |
| ethmagfight |Scaled number of rebel combatants or activists in ethnic war (range 0-4) | Integer | Categorical Ordinal |
| polity2 |Polity scale ranging from +10 (strongly democratic) to -10 (strongly autocratic) | Integer | Categorical Non-Ordinal |
| durable |Regime durabilty: the number of years since the most recent regime | Float | Categorical Ordinal |
| yip | Number of years head of state has been in power | Integer | Continuous |
| none | No violence occured | Float | Binary |
| eth | Indicates the occurence of an ethnic violence | Float | Binary |
| rev | Indicates the occurence of a revolutionary violence | Float | Binary |
| gen | Indicates the occurence of a genocide | Float | Binary |
| ethnic | Ethnic Fractionalization | Float | Continuous |
| linguistic | Linguistic Fractionalization|  Float | Continuous |
| religious | Religious Fractionalization | Float | Continuous |


A more detailed explanation of this project can be found on **[my website](http://lolalasisi.wixsite.com/mysite/data-to-coup)**

