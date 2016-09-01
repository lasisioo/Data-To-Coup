
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.axes as ax
get_ipython().magic(u'matplotlib inline')
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


# In[2]:

# Loading the necessary csv files
coup = pd.read_csv("../assets/csv/Coups.csv")
stab = pd.read_csv("../assets/csv/Politicalinstability2.csv")
polity = pd.read_csv("../assets/csv/polity.csv")
leaders = pd.read_csv("../assets/csv/leaders.csv")
urban = pd.read_csv("../assets/csv/urbanpop.csv")
gdp = pd.read_csv("../assets/csv/Realpercapitagdp.csv")
growth = pd.read_csv("../assets/csv/GDPgrowth.csv")


# In[3]:

# Summing scoup1(successful coups) and acoup1(attempted coups) to generate sumcoup
# I want a columns with binary variables that indicates coup (successful and attempted) or no coup
coup["sumcoup"] = coup["scoup1"] + coup["atcoup2"]

# To make a bianary column
coupbin = []
for i in coup["sumcoup"].tolist():
    if i > 0:
        coupbin.append(1)
    if i == 0:
        coupbin.append(0)
coup["Coup"] = coupbin


# In[4]:

# Dropping unnecessary columns
coup.drop(["scoup1", "atcoup2", "sumcoup"], axis=1, inplace=True)


# In[5]:

# To covert the variables in PTYPE to dummy variables 
dummies = pd.get_dummies(stab["PTYPE"])
stab = stab[["COUNTRY", "YEAR", "GENDEATHMAG", "ETHMAGFATAL", 
             "ETHMAGFIGHT", "REVMAGFATAL", "REVMAGFIGHT"]].join(dummies)

stab.columns = ["COUNTRY", "YEAR", "GENDEATHMAG", "ETHMAGFATAL", 
                "ETHMAGFIGHT", "REVMAGFATAL", "REVMAGFIGHT", "NONE",
                "ETH", "REV", "GEN"]


# In[6]:

# To avoid repetition of rows, I want to group the dataframes bq
aggMap = { "NONE" : "max", "ETH" : "max", "REV" : "max", 
           "GEN" : "max", "GENDEATHMAG" : "max", 
           "ETHMAGFATAL" : "max", "ETHMAGFIGHT" : "max", 
           "REVMAGFATAL" : "max", "REVMAGFIGHT" : "max" }

stabGrouped = stab.groupby(["COUNTRY","YEAR"]).agg(aggMap).reset_index()


# In[7]:

# Joining the political stabilty and coup tables 
coup1 = coup.merge(stabGrouped, how="inner", left_on=["country", "year"], right_on=["COUNTRY", "YEAR"])


# In[8]:

# dropping the unnecessary columns and setting all columns to lower case for easier manipulation in the future
coup1.drop(["COUNTRY", "YEAR"], axis=1, inplace=True)
coup1.columns = map(str.lower, coup1.columns)


# In[9]:

# Dropping unnecessary variables
polity.drop(["flag", "fragment", "democ", "autoc", "polity"], axis=1, inplace=True)


# In[10]:

# Merging polity table with coup1 table
coup2 = coup1.merge(polity, how="inner", left_on=["country", "year"], right_on=["country", "year"])


# In[11]:

# Merging coup2 and leaders
coup3 = coup2.merge(leaders, how="inner", left_on=["country", "year"], right_on=["country", "year"])


# In[12]:

# I want to change the shape of the table, so the year columns become rows
Col1 = urban.columns[0]
Col2 = urban.columns[1:].tolist()
urban1 = pd.melt(urban, id_vars=(Col1), var_name="year", value_vars=(Col2), value_name="urbanpop")
urban1.sort_values(["country", "year"], inplace=True)


# In[13]:

# Also changing the shape of the table here
Col3 = gdp.columns[0]
Col4 = gdp.columns[1:].tolist()
gdp1 = pd.melt(gdp, id_vars=(Col3), var_name="year", value_vars=(Col4), value_name="gdppercap")
gdp1.sort_values(["country", "year"], inplace=True)


# In[14]:

# Cleaning the values in the gdp1 table
gdp1['gdppercap'] = gdp1['gdppercap'].str.replace('$', '')
gdp1['gdppercap'] = gdp1['gdppercap'].str.replace(',', '')


# In[15]:

# Coverting the values to numeric
gdp1['gdppercap'] = gdp1['gdppercap'].convert_objects(convert_numeric=True)
gdp1.dtypes


# In[16]:

# Changing the shape of the table
Col5 = growth.columns[0]
Col6 = growth.columns[1:].tolist()
growth1 = pd.melt(growth, id_vars=(Col5), var_name="year", value_vars=(Col6), value_name="gdpgrowth")
growth1.sort_values(["country", "year"], inplace=True)


# In[17]:

# Converting to numeric
growth1['gdpgrowth'] = growth1['gdpgrowth'].convert_objects(convert_numeric=True)
growth1.dtypes


# In[18]:

# Joining tables again
Econ = gdp1.merge(growth1, how="inner", left_on=["country", "year"], right_on=["country", "year"])
Econ1 = Econ.merge(urban1, how="inner", left_on=["country", "year"], right_on=["country", "year"])


# In[19]:

# Convert the year column to numeric
Econ1["year"] = Econ1["year"].convert_objects(convert_numeric=True)


# In[20]:

# Final join!
DF = Econ1.merge(coup3, how="inner", left_on=["country", "year"], right_on=["country", "year"])


# In[21]:

# Incorporating a new table into the dataframe
# Importing the csv file as a list of lists
# Converting the list to a data dictionary, with the countries being the keys and ethinic, linguistic, and religious
# fractionalization respectively serving as the values

path = "../assets/csv/diversity.csv"
import csv

def read_file(path):
    with open(path, 'r') as f:
        diversity = [row for row in csv.reader(f.read().splitlines())]
    return diversity

diversity = read_file(path)
diversity = diversity[1:]
#print diversity

def function(i):return (i[0], i[1:])
div_dict = map(function, diversity)
divDict = dict(div_dict)


# In[22]:

# Matching the values from the dictionary to their respective countries

loc = DF["country"].tolist()


ethnicity = []
linguistic = []
religious = []


for i in loc:
    for country, ethnic in divDict.items():
        if i == country:
            ethnicity.append( ethnic[0] )
            linguistic.append( ethnic[1] )
            religious.append( ethnic[2] )


# In[23]:

#Assigning these values to their respective columns
DF["ethnic"] = ethnicity
DF["linguistic"] = linguistic
DF["religious"] = religious


# In[24]:

# Drop rows with missing values (since they are relatively few)
# Drop the year columns since I won't be needing it for my analysis
DF.dropna(axis=0, how="any", inplace=True)
DF.drop(["year"], axis=1, inplace=True)


# In[25]:

# Converting year column to numeric
DF[["ethnic", "linguistic", "religious"]] = DF[["ethnic", "linguistic", "religious"]]                                             .convert_objects(convert_numeric=True)


# In[26]:

DF = DF[DF.ethmagfatal != 9]
DF = DF[DF.ethmagfight != 9]
DF = DF[DF.revmagfatal != 9]
DF = DF[DF.revmagfight != 9]


# <h1>Data Dictionary</h1>
# 
# 
# Variable|Description|Data Type|Variable Type
# --|--
# country|Country Name|Object|Unique
# coup|1 = Coup(successful and attempted), 0 = No coup|Integer|Binary
# gdppercap|GDP per capita|Float|Continuous 
# gdpgrowth|GDP growth rate|Float|Continuous
# urbanpop|Percentage of urban population|Float|Continuous
# gendeathmag|Scaled annual number of deaths (range 0-5.0)|Float|Categorical Ordinal
# revmagfight|Scaled annual number of fatalities related to revolutionary fighting (range 0-4)|Integer|Categorical Ordinal
# ethmagfatal|Scaled annual number of fatalities related to ethnic fighting (range 0-4)|Integer|Categorical Ordinal
# revmagfatal|Scaled number of rebel combatants or activists in revolutionary war (range 0-4)|Integer|Categorical Ordinal
# ethmagfight|Scaled number of rebel combatants or activists in ethnic war (range 0-4)|Integer|Categorical Ordinal
# polity2|Polity scale ranging from +10 (strongly democratic) to -10 (strongly autocratic)|Integer|Categorical Non-Ordinal
# durable|Regime durabilty: the number of years since the most recent regime|Float|Categorical Ordinal
# yip|Number of years head of state has been in power|Integer|Continuous
# none|No violence occured|Float|Binary
# eth|Indicates the occurence of an ethnic violence|Float|Binary
# rev|Indicates the occurence of a revolutionary violence|Float|Binary
# gen|Indicates the occurence of a genocide|Float|Binary
# ethnic| Ethnic Fractionalization|Float|Continuous
# linguistic|Linguistic Fractionalization|Float|Continuous
# religious| Religious Fractionalization|Float|Continuous

# In[29]:

# To observe the differences in distribution of feactures for coup and no coup
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12),       (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(nrows=6, ncols=3,         sharey=False, sharex=True, figsize=(20,30))
fig.suptitle("Feature Distributions for Coup and No Coup", size=50)

sns.set_style("whitegrid")
sns.violinplot(x='coup', y="gdppercap", data=DF, ax=ax1)
ax1.set_xlabel("Coup", fontsize=15)
ax1.set_ylabel("GDP/Capita", fontsize=15)  
    
sns.violinplot(x='coup', y="gdpgrowth", data=DF, ax=ax2)
ax2.set_xlabel("Coup", fontsize=15)
ax2.set_ylabel("GDP Growth", fontsize=15)

sns.violinplot(x='coup', y="urbanpop", data=DF, ax=ax3)
ax3.set_xlabel("Coup", fontsize=15)
ax3.set_ylabel("Urban Pop.", fontsize=15)

sns.violinplot(x='coup', y="none", data=DF, ax=ax4)
ax4.set_xlabel("Coup", fontsize=15)
ax4.set_ylabel("None", fontsize=15)

sns.violinplot(x='coup', y="revmagfight", data=DF, ax=ax5)
ax5.set_xlabel("Coup", fontsize=15)
ax5.set_ylabel("RevMagFight", fontsize=15)

sns.violinplot(x='coup', y="gendeathmag", data=DF, ax=ax6)
ax6.set_xlabel("Coup", fontsize=15)
ax6.set_ylabel("GenDeathMag", fontsize=15)

sns.violinplot(x='coup', y="ethmagfatal", data=DF, ax=ax7)
ax7.set_xlabel("Coup", fontsize=15)
ax7.set_ylabel("EthMagFatal", fontsize=15)

sns.violinplot(x='coup', y="revmagfatal", data=DF, ax=ax8)
ax8.set_xlabel("Coup", fontsize=15)
ax8.set_ylabel("RevMagFatal", fontsize=15)

sns.violinplot(x='coup', y="eth", data=DF, ax=ax9)
ax9.set_xlabel("Coup", fontsize=15)
ax9.set_ylabel("Eth", fontsize=15)

sns.violinplot(x='coup', y="ethmagfight", data=DF, ax=ax10)
ax10.set_xlabel("Coup", fontsize=15)
ax10.set_ylabel("EthMagFight", fontsize=15)

sns.violinplot(x='coup', y="rev", data=DF, ax=ax11)
ax11.set_xlabel("Coup", fontsize=15)
ax11.set_ylabel("Rev", fontsize=15)

sns.violinplot(x='coup', y="gen", data=DF, ax=ax12)
ax12.set_xlabel("Coup", fontsize=15)
ax12.set_ylabel("Gen", fontsize=15)

sns.violinplot(x='coup', y="polity2", data=DF, ax=ax13)
ax13.set_xlabel("Coup", fontsize=15)
ax13.set_ylabel("Polity", fontsize=15)

sns.violinplot(x='coup', y="durable", data=DF, ax=ax14)
ax14.set_xlabel("Coup", fontsize=15)
ax14.set_ylabel("Durable", fontsize=15)

sns.violinplot(x='coup', y="linguistic", data=DF, ax=ax15)
ax15.set_xlabel("Coup", fontsize=15)
ax15.set_ylabel("Linguistic", fontsize=15)

sns.violinplot(x='coup', y="yip", data=DF, ax=ax16)
ax16.set_xlabel("Coup", fontsize=15)
ax16.set_ylabel("YIP", fontsize=15)

sns.violinplot(x='coup', y="ethnic", data=DF, ax=ax17)
ax17.set_xlabel("Coup", fontsize=15)
ax17.set_ylabel("Ethnic", fontsize=15)

sns.violinplot(x='coup', y="religious", data=DF, ax=ax18)
ax18.set_xlabel("Coup", fontsize=15)
ax18.set_ylabel("Religious", fontsize=15)

fig.subplots_adjust(hspace=.3, wspace=.3)


# In[30]:

# To split the dataframe into one with instances of a coup and another with instances of no coup
acoup = DF.loc[DF.coup==1]
nocoup = DF.loc[DF.coup==0]


# In[31]:

# To test for statistical differences in the means of features for coup and no coup
features = ["gdppercap", "gdpgrowth", "urbanpop", "none", "revmagfight", 
            "gendeathmag", "ethmagfatal", "revmagfatal", "eth", 
            "ethmagfight", "rev", "gen", "polity2", "durable", 
            "yip", "ethnic", "linguistic", "religious"]

for i in features:
    print ttest_ind(acoup[i], nocoup[i])


# ## Data Exploration

# In[32]:

gBPolity = DF.groupby("polity2").mean()
gBPolity


# In[33]:

DF.groupby("coup").mean()


# In[46]:

# Data dictionary indicating regions around the world (keys), and countries that belong in each region (values).
worldMap = {
            "southAmerica":     ["Argentina", "Chile", "Colombia", "Peru"],
            "centralAmerica":   ["El Salvador", "Guatemala", "Mexico", "Nicaragua"],
            "caribbean":        ["Cuba", "Dominican Republic", ],
            "northernEurope":   ["United Kingdom"],
            "southernEurope":   ["Albania", "Croatia"],
            "easternEurope":    ["Hungary", "Moldova", "Romania", "Russia", "Ukraine"],
            "centralAsia":      ["Tajikistan"],
            "southEasternAsia": ["Cambodia", "Indonesia", "Laos", "Philippines", "Thailand"],
            "southernAsia":     ["Afghanistan", "Bangladesh", "India", "Iran", "Nepal","Pakistan", 
                                 "Sri Lanka"],
            "easternAsia":      ["China"],
            "westernAsia":      ["Azerbaijan", "Cyprus", "Georgia", "Iraq", "Israel", "Lebanon", 
                                 "Oman", "Syria", "Turkey"],
            "oceania":          ["Papua New Guinea"],
            "northernAfrica":   ["Algeria", "Egypt", "Libya", "Morocco", "Sudan"],
            "southernAfrica":   ["South Africa"],
            "easternAfrica":    ["Burundi", "Djibouti", "Ethiopia", "Kenya", "Mozambique", "Rwanda",
                                 "Uganda", "Zambia", "Zimbabwe"],
            "westernAfrica":    ["Guinea", "Guinea-Bissau", "Liberia", "Mali", "Nigeria", "Senegal", 
                                 "Sierra Leone"],
            "middleAfrica":     ["Angola", "Central African Republic", "Chad", "DR Congo", 
                                 "Equatorial Guinea"]
           }


# In[35]:

# Creating a dummy variabes for each region.
def binCountry(country, worldMap):
    for region, countryList in worldMap.items():
        if country in countryList:
            return region
        

DF["CountryBin"] = DF.country.apply( lambda country: binCountry(country, worldMap) )
dummies = pd.get_dummies( DF["CountryBin"] )
DF2 = DF.join(dummies)
DF2.drop(["country", "CountryBin"], axis=1, inplace=True)


# In[36]:

# Correlation map between all variables.
sns.heatmap(DF2.corr())


# In[37]:

# Creating the target variable and the features.
y = DF2["coup"]
X = DF2.drop(["coup"], axis=1)


# In[38]:

from sklearn.cross_validation import KFold,cross_val_score,train_test_split,cross_val_predict
from sklearn.metrics import r2_score,accuracy_score,precision_score,recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot


# In[47]:

# Splitting the detaset into a train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Random Forest Classifier

# In[40]:

# GridSearch

# rf = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True) 

# rfparam_grid = { 
#                 'n_estimators': [25, 50,75,100,125,150,200],
#                 'criterion': ["gini", "entropy"],
#                 'max_features': [None, 'sqrt', 'log2'],
#                 'min_samples_split':[1,2,3,4,5,6]
#                }
    
# CV_rf = GridSearchCV(estimator=rf, param_grid=rfparam_grid, cv=5)
# CV_rf.fit(X_train, y_train)
# print CV_rf.best_params_


# In[68]:

# Parameters were selected based on the results from the gridsearch.
cv = KFold(len(y_train), shuffle=False) 
print cv
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features="sqrt", max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
            oob_score=True, random_state=20, verbose=0, warm_start=False)
rfScore = cross_val_score(rf, X_train, y_train, cv=cv, n_jobs=-1)
print "Regular Random Forest scores are:", rfScore
print "Regular Random Forest average score is:", rfScore.mean()


# In[42]:

# Fitting the variables in the model 
rfModel = rf.fit(X_train, y_train)


# In[43]:

# Using the model from the train set to predict the target for the test set.
rfPredicted = rfModel.predict(X_test)
rfProbs = rfModel.predict_proba(X_test)


# In[44]:

# Creating a dataframe with actual and predicted coup results, as wel as probability of coup or no coup.
rfPredictions = pd.DataFrame()
rfPredictions["Actual"] = y_test
rfPredictions["Predicted"] = rfPredicted
rfPredictions["ProbsPos"], rfPredictions["ProbsNeg"] = zip(*rfProbs)


# In[45]:

# Deriving the threshold for the highest ROC score.
newList = []
newList2 = []
for i in np.arange(0,1,0.01):
    rfPredictions["ThreshPred"] = ([0 if x < i else 1 for x in rfPredictions['ProbsNeg']])
    newList.append(roc_auc_score(rfPredictions["Actual"], rfPredictions["ThreshPred"]))
    newList2.append(i)
d = zip(newList, newList2)
max(d)


# In[49]:

# Using the threshold to predict whehther or nor a coup occurred
rfThreshold = max(d)[1]
rfPredictions["ThreshPred"] = ([0 if x < rfThreshold else 1 for x in rfPredictions['ProbsNeg']])


# In[50]:

print 'accuracy score:', accuracy_score(rfPredictions["Actual"], rfPredictions["ThreshPred"])
print 'ROC Score:', roc_auc_score(rfPredictions["Actual"], rfPredictions["ThreshPred"])
print 'precision score:', precision_score(rfPredictions['Actual'],rfPredictions['ThreshPred'])
print 'recall score:', recall_score(rfPredictions['Actual'],rfPredictions['ThreshPred'])


# In[51]:

# Confusion Matrix
print pd.crosstab(rfPredictions["Actual"], rfPredictions["ThreshPred"], rownames=["Actual"])


# ## AdaBoost Classifier

# In[52]:

# GridSearch

# ab = AdaBoostClassifier() 

# abparam_grid = { 
#                 "n_estimators": [10,15,20,25,30],
#                 "learning_rate": [1.0,2.0,3.0,4.0,5.0,6.0],
#                 "algorithm": ["SAMME", "SAMME.R"]
#                }

# CV_ab= GridSearchCV(estimator=ab, param_grid=abparam_grid, cv=5)
# CV_ab.fit(X_train, y_train)
# print CV_ab.best_params_
# print CV_ab.best_estimator_


# In[53]:

# Parameters were selected based on the results from the gridsearch.
cv = KFold(len(y_train), shuffle=False) 
print cv
ab = AdaBoostClassifier(algorithm='SAMME', base_estimator=None,
          learning_rate=2.0, n_estimators=15, random_state=20)

abScore = cross_val_score(ab, X_train, y_train, cv=cv,n_jobs=1)
print "Adaboost Decision Tree scores are:", abScore
print "Adaboost Decision Tree average score is:", abScore.mean()


# In[54]:

# Fitting the variables in the model 
abModel = ab.fit(X_train, y_train)


# In[55]:

# Using the model from the train set to predict the target for the test set.
abPredicted = abModel.predict(X_test)
abProbs = abModel.predict_proba(X_test)


# In[56]:

# Creating a dataframe with actual and predicted coup results, as wel as probability of coup or no coup.
abPredictions = pd.DataFrame()
abPredictions["Actual"] = y_test
abPredictions["Predicted"] = abPredicted
abPredictions["ProbsPos"], abPredictions["ProbsNeg"] = zip(*abProbs)


# In[57]:

# Deriving the threshold for the highest ROC score.
newList3 = []
newList4 = []
for i in np.arange(0,1,0.01):
    abPredictions["ThreshPred"] = ([0 if x < i else 1 for x in abPredictions['ProbsNeg']])
    newList3.append(roc_auc_score(abPredictions["Actual"], abPredictions["ThreshPred"]))
    newList4.append(i)
e = zip(newList3, newList4)
max(e)


# In[58]:

# Using the threshold to predict whehther or nor a coup occurred
abThreshold = max(e)[1]
abPredictions["ThreshPred"] = ([0 if x < abThreshold else 1 for x in abPredictions['ProbsNeg']])


# In[59]:

print 'accuracy score:', accuracy_score(abPredictions["Actual"], abPredictions["ThreshPred"])
print 'ROC Score:', roc_auc_score(abPredictions["Actual"], abPredictions["ThreshPred"])
print 'precision score:', precision_score(abPredictions['Actual'],abPredictions['ThreshPred'])
print 'recall score:', recall_score(abPredictions['Actual'],abPredictions['ThreshPred'])


# In[60]:

#Confusion Matrix
print pd.crosstab(abPredictions["Actual"], abPredictions["ThreshPred"], rownames=["Actual"])


# In[61]:

# Compute ROC curve and ROC area for Random Forest
rffpr = dict()
rftpr = dict()
rfroc_auc = dict()
rffpr, rftpr, _ = roc_curve(y_test, rfPredictions.ProbsNeg)
rfroc_auc = auc(rffpr, rftpr)

# Compute ROC curve and ROC area for Adaboost
abfpr = dict()
abtpr = dict()
abroc_auc = dict()
abfpr, abtpr, _ = roc_curve(y_test, abPredictions.ProbsNeg)
abroc_auc = auc(abfpr, abtpr)


# Plot of a ROC curve 
plt.figure(figsize=(15,15))
plt.plot(rffpr,rftpr,label='Rf AUC = %0.2f' % rfroc_auc)
plt.plot(abfpr,abtpr,label='Ab AUC = %0.2f' % abroc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic\n', fontsize=30)
plt.legend(loc="lower right", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.show()


# ## Random Forest with Feature Selection - Robustness

# In[62]:

# Averaging feature importances across trees in random forest model.
all(rf.feature_importances_ == np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0))

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]
feature_names = X_train.columns


# In[63]:

# Graph displaying feature importance.
plt.figure(figsize=(20,10))
plt.title("Feature Importances\n", fontsize = 30)
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90, fontsize = 20)
plt.xlim([-1, X_train.shape[1]])
plt.yticks(fontsize=20)


# In[64]:

# New X variables based on feature importance.
X_new = DF2[["gdppercap", "urbanpop", "gdpgrowth", "durable", "yip",
            "polity2", "ethnic", "religious", "linguistic"]]


# In[65]:

# Splitting the dataset.
X_newtrain, X_newtest, y_newtrain, y_newtest = train_test_split(X_new, y, test_size=0.3, random_state=0)


# In[66]:

# Grid Search

# rf2 = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True) 

# rf2param_grid = { 
#                 'n_estimators': [50,75,100,125,150],
#                 'criterion': ["gini", "entropy"],
#                 'max_features': [None, 'sqrt', 'log2'],
#                 'min_samples_split':[1,2,3,4,5,6]
#                 }
    
# CV_rf2 = GridSearchCV(estimator=rf2, param_grid=rf2param_grid, cv=5)
# CV_rf2.fit(X_newtrain, y_newtrain)
# print CV_rf2.best_params_


# In[67]:

# Parameters were selected based on the results from the gridsearch.
cv2 = KFold(len(y_newtrain), shuffle=False) 
print cv2
rf2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features="sqrt", max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
            oob_score=True, random_state=20, verbose=0, warm_start=False)
rfScore = cross_val_score(rf, X_train, y_train, cv=cv, n_jobs=-1)
print "Regular Decision Tree scores are:", rfScore
print "Regular Decision Tree average score is:", rfScore.mean()


# In[69]:

# Fitting the variables in the model. 
rf2Model = rf2.fit(X_newtrain, y_newtrain)


# In[70]:

# Using the model from the train set to predict the target for the test set.
rf2Predicted = rf2Model.predict(X_newtest)
rf2Probs = rf2Model.predict_proba(X_newtest)


# In[71]:

# Creating a dataframe with actual and predicted coup results, as wel as probability of coup or no coup.
rf2Predictions = pd.DataFrame()
rf2Predictions["Actual"] = y_newtest
rf2Predictions["Predicted"] = rf2Predicted
rf2Predictions["ProbsPos"], rf2Predictions["ProbsNeg"] = zip(*rf2Probs)


# In[72]:

# Deriving the threshold for the highest ROC score.
newList5 = []
newList6 = []
for i in np.arange(0,1,0.01):
    rf2Predictions["ThreshPred"] = ([0 if x < i else 1 for x in rf2Predictions['ProbsNeg']])
    newList5.append(roc_auc_score(rf2Predictions["Actual"], rf2Predictions["ThreshPred"]))
    newList6.append(i)
f = zip(newList5, newList6)
max(f)


# In[73]:

# Using the threshold to predict whehther or nor a coup occurred
rf2Threshold = max(f)[1]
rf2Predictions["ThreshPred"] = [0 if x < rf2Threshold else 1 for x in rf2Predictions['ProbsNeg']]


# In[75]:

print 'accuracy score:', accuracy_score(rf2Predictions["Actual"], rf2Predictions["ThreshPred"])
print 'ROC Score:', roc_auc_score(rf2Predictions["Actual"], rf2Predictions["ThreshPred"])
print 'precision score:', precision_score(rf2Predictions['Actual'],rf2Predictions['ThreshPred'])
print 'recall score:', recall_score(rf2Predictions['Actual'],rf2Predictions['ThreshPred'])
print 'accuracy score:',accuracy_score(rf2Predictions['Actual'],rf2Predictions['ThreshPred'])


# In[74]:

# Confusion Matrix
print pd.crosstab(rf2Predictions["Actual"], rf2Predictions["ThreshPred"], rownames=["Actual"])


# In[76]:

# Compute ROC curve and ROC area for Random Forest (with feature selection)
rffpr2 = dict()
rftpr2 = dict()
rfroc_auc2 = dict()
rffpr2, rftpr2, _ = roc_curve(y_newtest, rf2Predictions.ProbsNeg)
rfroc_auc2 = auc(rffpr2, rftpr2)

# Plot of a ROC curve 
plt.figure(figsize=(15,15))
plt.plot(rffpr2,rftpr2,label='Rf AUC = %0.2f' % rfroc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic\n', fontsize=30)
plt.legend(loc="lower right", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.show()


# ## Logistic Regression

# In[77]:

from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


# In[78]:

y1, X1 = dmatrices("coup ~ gdppercap + gdpgrowth + urbanpop + none +                     revmagfight + gendeathmag + ethmagfatal + revmagfatal +                     eth + ethmagfight + rev + gen + polity2 + durable + yip +                     ethnic + linguistic + religious + caribbean + centralAmerica +                     centralAsia + easternAfrica + easternAsia + easternEurope +                     middleAfrica + northernAfrica + northernEurope + oceania + southAmerica +                     southEasternAsia + southernAfrica + southernAsia + southernEurope +                     westernAfrica", DF2, return_type="dataframe")


# In[79]:

y1 = np.ravel(y1)


# In[80]:

lr = LogisticRegression()
lrModel = lr.fit(X1,y1)


# In[81]:

coeffName =  X1.columns.tolist()
coeff =  lrModel.coef_[0]
coeffs = pd.DataFrame(zip(coeffName, coeff), columns=["CoeffName", "Coeff"])
coeffs.head()


# In[82]:

X_norm =  StandardScaler().fit_transform(X1)

lrModel2 = lr.fit(X_norm, y1)
coeff2 = lrModel2.coef_[0]
coeffs2 = pd.DataFrame(zip(coeffName, coeff2), columns=["CoeffName", "Coeff"])
coeffs3 = coeffs2
coeffs3["Coeff"] = abs(coeffs3["Coeff"])
coeffs3.sort("Coeff", ascending = False, inplace=True)


# In[83]:

y2, X2 = dmatrices("coup ~ gdpgrowth + urbanpop + none +                      durable + polity2 + gdppercap + ethnic +                      linguistic +  + caribbean + easternAfrica +                      easternAsia + easternEurope + southernAfrica +                      southernAsia", DF2, return_type="dataframe")


# In[84]:

y2 = np.ravel(y2)


# In[85]:

lrModel2 = lr.fit(X2, y2)
# logit = sm.Logit(X2, y2)
# result = logit.fit()
# result.summary()
lrModel2.coef_


# In[86]:

coeffName2 =  X2.columns.tolist()
coeff2 =  np.exp(lrModel2.coef_[0])
coeffs2 = pd.DataFrame(zip(coeffName2, coeff2), columns=["CoeffName", "Coeff"])
coeffs2


# In[ ]:



