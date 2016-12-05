# Data To Coup


A Data Science Project by [Dami Lasisi](http://lolalasisi.wixsite.com/mysite) in minor collaboration with [Morgan Murrah](https://www.github.com/airbr) providing some Web Development on the front end display/interaction.

 <b>Problem Statement:</b> Considering social and political instabilities that have occurred in recent times, I want predict the likelihood of a coup d'etat within a country, given specific economic, political and social factors.

- Economic factors: income per capita, growth rate
- Social factors: accounts of ethnic violence, religious, linguistic, and ethnic fractionalization, fatalities from 
  violence, urbanization
- Political factors: polity score, length of stay for incumbent head of state

[A more detailed analyis of this project is hosted at the following website](http://lolalasisi.wixsite.com/mysite/data-to-coup)

--

# Quickstart / Links:

#### [Jupyter Notebook](https://jupyter.org/).
If you have Jupyter Netbook installed you should be able to view it remotely hosted at the following URL: TBD

#### Local machine.
Follow the below instructions to install the necessary dependencies for viewing the notebook locally:

<!--`-> Visualizations.`
See the source code behind the visualizations in the following snippets-->

## [Installing Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)

### Local machine pre-requisites

* Python 2.7 
	* Check for default python install 
	* Installing Python 2.7
* Anaconda with following python libraries installed using anaconda:
	* Manually from terminal with conda packages (if not included by default in your environment):
		1. `conda install numpy`
		2. `conda install pandas`
		3. `conda install scikit-learn`
		4. `conda install matplotlib`
		5. `conda install patsy`
		6. `conda install seaborn`
		7. `conda install pydot`

### Data Models used:
- Random Forest
- AdaBoost
- Logistic Regression

<!--<b>Problem Statement:</b> Considering social and political instabilities that have occurred in recent times, I want predict the likelihood of a coup d'etat within a country, given specific economic, political and social factors.

- Economic factors: income per capita, growth rate
- Social factors: accounts of ethnic violence, religious, linguistic, and ethnic fractionalization, fatalities from 
  violence, urbanization
- Political factors: polity score, length of stay for incumbent head of state
-->

# Data Dictionary:

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

