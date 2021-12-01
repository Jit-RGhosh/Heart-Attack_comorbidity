#!/usr/bin/env python
# coding: utf-8

# # Creating Readable File at Python by Importing Data Sheet Available in csv format

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Preparing a function for evaluating the models

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2), 
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict

Heart_fail1 = pd.read_csv("E:/Python/Python - Decodr/DecodR_Class/October - 2021/Projects/Heart Attack Ratio/heart_failure_clinical_records_dataset.csv")
Heart_fail1.shape


# In[ ]:





# In[117]:


print(Heart_fail1)


# # Description of Data Sheet

# In[118]:


Heart_fail1.info()


# ## As you see No data cell contain null value hence we consider Data cleaning may not required.
# ## There is two type of numaric data available in this segment - 1. Intiger Data; 2. Float Data

# # Determine Name of the Column

# In[119]:


Heart_fail1.columns


# ## Aboves are name of the colums available in the given data sheet.

# # Checking for Null Value in Data Sheet

# In[120]:


Heart_fail1.isnull()


# ## Above Table shows there is no null value pressent in the given sheet

# In[121]:


Heart_fail1


# # EDA

# In[122]:


sns.pairplot(Heart_fail1)


# # Showing Unique Value in Each Collumn of given Data Set

# In[123]:


for i in Heart_fail1.columns:
    print (i,Heart_fail1[i].nunique())


# In[124]:


sns.countplot(data=Heart_fail1, x='anaemia')
plt.show()
print(Heart_fail1['anaemia'].value_counts())


# ## 129 People Who have Anaemia has experience Heart Attack
# ## 170 People who dont have anaemia but still experience Heart attack

# In[125]:


pd.crosstab(Heart_fail1.anaemia  ,Heart_fail1.DEATH_EVENT).plot(kind='bar')
plt.title('Death Event as per Anaemia')
plt.xlabel('Anaemia')
plt.ylabel('Death')
plt.show()
print(Heart_fail1['anaemia'].value_counts())


# In[126]:


anaemia_death = Heart_fail1[Heart_fail1['DEATH_EVENT']==1][Heart_fail1['anaemia']==1]
anaemia_death.shape[0]


# In[127]:


print("Percentage of people who died and have Anaemia:", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["anaemia"] == 1].value_counts(normalize = True)[1]*100)


# # Approximatly 1/3 rd of person who died in heart attack can have Anaemia as a co-morbidity factor

# In[128]:


plt.figure(figsize = (10,10)) #(W,h)
sns.countplot(data=Heart_fail1, x='creatinine_phosphokinase')
plt.show()
print(Heart_fail1['creatinine_phosphokinase'].value_counts())


# In[129]:


pd.crosstab(Heart_fail1.creatinine_phosphokinase  ,Heart_fail1.DEATH_EVENT).plot(kind='bar')
plt.title('Death Event as per ')
plt.xlabel('creatinine_phosphokinase')
plt.ylabel('Death')
plt.show()
print(Heart_fail1['DEATH_EVENT'].value_counts())


# # Effect of Blood Pressure on Heart Attack

# In[130]:


sns.countplot(data=Heart_fail1, x='high_blood_pressure')
plt.show()
print(Heart_fail1['high_blood_pressure'].value_counts())


# In[131]:


pd.crosstab(Heart_fail1.high_blood_pressure  ,Heart_fail1.DEATH_EVENT).plot(kind='bar')
plt.title('Death Event as per Anaemia')
plt.xlabel('high_blood_pressure')
plt.ylabel('Death')
plt.show()
print(Heart_fail1['DEATH_EVENT'].value_counts())


# In[132]:


print("Percentage of people who died and have High Blood Pressure:", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["high_blood_pressure"] == 1].value_counts(normalize = True)[1]*100)


# ## The above data shows that more then 37.14% people Die in Heart Attack if they have High BP

# # Gender Ratio of Heart attack 

# In[133]:


sns.countplot(data=Heart_fail1, x='sex')
plt.show()
print(Heart_fail1['sex'].value_counts())


# In[134]:


pd.crosstab(Heart_fail1.sex  ,Heart_fail1.DEATH_EVENT).plot(kind='bar')
plt.title('Death Event as per Anaemia')
plt.xlabel('Gendar')
plt.ylabel('Death')
plt.show()
print(Heart_fail1['DEATH_EVENT'].value_counts())


# In[135]:


print("Percentage of Female died :", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["sex"] == 0].value_counts(normalize = True)[1]*100)


# In[136]:


print("Percentage of Male died :", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["sex"] == 1].value_counts(normalize = True)[1]*100)


# ## As per sample data of the Hospital Mortality rate of Female is slightly higher then the male.

# # Effect of Smotking in Heart Attack

# In[137]:


sns.countplot(data=Heart_fail1, x='smoking')
plt.show()
print(Heart_fail1['smoking'].value_counts())


# In[138]:


pd.crosstab(Heart_fail1.smoking  ,Heart_fail1.DEATH_EVENT).plot(kind='bar')
plt.title('Death Event as per Anaemia')
plt.xlabel('Smoking')
plt.ylabel('Death')
plt.show()


# In[139]:


print("Percentage of Non-Smoker died :", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["smoking"] == 0].value_counts(normalize = True)[1]*100)


# In[140]:


print("Percentage of Smoker died :", 
      Heart_fail1["DEATH_EVENT"][Heart_fail1["smoking"] == 1].value_counts(normalize = True)[1]*100)


# ## Data Fact clearly shows death rate of non-smoker is higher the the Smoker hence Smoking dont have much effect on Heart Attack.

# In[141]:


#Looking for and visualizing correlations with correlation matrix on raw data
fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(Heart_fail1.corr(), annot=True, fmt='.2f')
plt.show()


# In[142]:


#Plotting the highest correlations
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,20))
red_dot = mlines.Line2D([], [], color='darkred', marker='o', linestyle='None',
                          markersize=5, label='Failure')
blue_dot = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None',
                          markersize=5, label='No Failure')
scatter1 = ax0.scatter(x=Heart_fail1.age, y=Heart_fail1.ejection_fraction, c=Heart_fail1.DEATH_EVENT, zorder=3, cmap="coolwarm")
ax0.set(title = "Heart Failure and Ejection Fraction", ylabel="Ejection Fraction, %")
ax0.grid(color="lightgrey", zorder=0)
ax0.legend(handles=[blue_dot, red_dot])
scatter2 = ax1.scatter(x=Heart_fail1.age, y=Heart_fail1.serum_creatinine, c=Heart_fail1.DEATH_EVENT, zorder=3, cmap="coolwarm")
ax1.set(title = "Heart Failure and Creatinine Level", ylabel="Creatinine Level, mg/dL")
ax1.grid(color="lightgrey", zorder=0)
ax1.legend(handles=[blue_dot, red_dot])
scatter3 = ax2.scatter(x=Heart_fail1.age, y=Heart_fail1.serum_sodium, c=Heart_fail1.DEATH_EVENT, zorder=3, cmap="coolwarm")
ax2.set(title = "Heart Failure and Sodium Level", xlabel="Age", ylabel="Sodium Level, mEq/L")
ax2.grid(color="lightgrey", zorder=0)
ax2.legend(handles=[blue_dot, red_dot])
scatter4 = ax3.scatter(x=Heart_fail1.age, y=Heart_fail1.time, c=Heart_fail1.DEATH_EVENT, zorder=3, cmap="coolwarm")
ax3.set(title = "Heart Failure and Follow-up period", xlabel="Age", ylabel="Follow-up period (days)")
ax3.grid(color="lightgrey", zorder=0)
ax3.legend(handles=[blue_dot, red_dot]);


# In[143]:


# The data seems to be sorted on time column. Let's correct it

data_schuffled = Heart_fail1.sample(frac = 1)
X = data_schuffled.drop("DEATH_EVENT", axis=1)
y = data_schuffled.DEATH_EVENT


# In[144]:


# Splitting in test and train sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[145]:


# Scaling (normalizing) X
scaler=StandardScaler()
scaler.fit(X)
X_train_n = scaler.transform(X_train)
X_test_n = scaler.transform(X_test)


# In[146]:


# Declaring models
models = {"KNN": KNeighborsClassifier(),
          "LogisticReg": LogisticRegression(),
          "RandomForest": RandomForestClassifier()}

# A function to select the best model
def find_best_model(models, X_train, X_test, y_train, y_test):
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_test, y_test)
    return scores


# In[147]:


# Let's see which algorithm performs best
scores = find_best_model(models=models, X_train=X_train_n, X_test=X_test_n, y_train=y_train, y_test=y_test)
scores


# # Seems RandomForest work better. lets tune and check score

# In[148]:


# Let's tune the LogReg
log_reg_grid = {"C": np.logspace(-4, 4, 40),
                "solver": ["liblinear"]}
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=40,
                                verbose=True)

# Fit random hyperparameter search model and show the score
rs_log_reg.fit(X_train_n, y_train)
rs_log_reg.score(X_test_n, y_test)


# In[149]:


rs_log_reg.best_params_


# In[150]:


# Let's cross-validate this model
rs_best_n = LogisticRegression(C = 0.011253355826007646, solver="liblinear")
rs_best_n.fit(X_train_n, y_train)
cross_val_score(rs_best_n, scaler.transform(X), y, cv=5)


# In[151]:


# O/p difference means model is not so perfect


# In[152]:


# Making predictions
y_preds_n = rs_best_n.predict(X_test_n)


# In[153]:


# Evaluating the model and plotting a confusion matrix
evaluate_preds(y_test, y_preds_n);
conf_mat = confusion_matrix(y_test, y_preds_n)
sns.heatmap(conf_mat/np.sum(conf_mat), fmt='.2%', annot=True, cbar=False,yticklabels=["No Failure", "Failure"],
    xticklabels=["Predicted No Failure", "Predicted Failure"]);


# # 65% predicting precision may not good when it concern about heart. lets check with RandomForest model

# In[154]:


# Setting, fitting the model, getting R2
RFCn = RandomForestClassifier()
RFCn.fit(X_train_n, y_train)
RFCn.score(X_test_n, y_test)


# In[155]:


# Cross-Validating the model
cross_val_score(RFCn, scaler.transform(X), y, cv=5)


# In[156]:


#less veriation model seems work well. lets check further 


# In[157]:


# Let's tune the RandomForest
grid = {"n_estimators": [10, 50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]}

clf = RandomForestClassifier(n_jobs=1)
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=40, 
                            cv=5)

# Fit random hyperparameter search model
rs_clf.fit(X_train_n, y_train)
rs_clf.score(X_test_n, y_test)


# In[158]:


rs_clf.best_params_


# In[159]:


clf_best = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, min_samples_split=6, max_features="auto", max_depth=10)
clf_best.fit(X_train_n, y_train)
clf_best.score(X_test_n, y_test)


# In[160]:


cross_val_score(clf_best, scaler.transform(X), y, cv=5).mean()


# In[161]:


# Let's make predictions
y_preds_n = clf_best.predict(X_test_n)


# In[162]:


# Evaluating and confisuon matrix
evaluate_preds(y_test, y_preds_n);
conf_mat = confusion_matrix(y_test, y_preds_n)
sns.heatmap(conf_mat/np.sum(conf_mat), fmt='.2%', annot=True, cbar=False,yticklabels=["No Failure", "Failure"],
    xticklabels=["Predicted No Failure", "Predicted Failure"]);


# In[163]:


# Model is better then previous one precision rate just below 80%. Let increase subset by 30%


# In[164]:


X = data_schuffled.drop("DEATH_EVENT", axis=1)
y = data_schuffled.DEATH_EVENT
np.random.seed(42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size=0.3)
clf_best.fit(scaler.transform(X_train1), y_train1)
print(f"Rsquared: {clf_best.score(scaler.transform(X_test1), y_test1)}, Cross-validated Rsquared: {cross_val_score(clf_best, scaler.transform(X), y, cv=5).mean()}")


# In[166]:


y_preds1 = clf_best.predict(scaler.transform(X_test1))
evaluate_preds(y_test1, y_preds1);
conf_mat = confusion_matrix(y_test1, y_preds1)
sns.heatmap(conf_mat/np.sum(conf_mat), fmt='.2%', annot=True, cbar=False,yticklabels=["No Failure", "Failure"],
    xticklabels=["Predicted No Failure", "Predicted Failure"]);


# # As we see that this model gives more then 80% correctness - we may consider this model work fine.

# In[111]:


# Different cross validated result, let find which row predicted badly.


# In[168]:


modelled_data = pd.DataFrame(X_test1.reset_index(drop=True), columns=Heart_fail1.drop("DEATH_EVENT", axis=1).columns)
modelled_data["real_death"]=y_test1.reset_index(drop=True)
modelled_data["predicted_death"]=y_preds1

# Make a column with badly predicted outcomes
modelled_data["false"] = abs(modelled_data.real_death - modelled_data.predicted_death)
modelled_data.head()


# In[169]:


# Let's compare the mean values of all parameters for good and bad predictions
modelled_data[modelled_data.false==1].mean()


# In[170]:


modelled_data[modelled_data.false==0].mean()


# In[171]:


# Lets Plot Both Good and Bad Model


# In[172]:


red_dot = mlines.Line2D([], [], color='darkred', marker='o', linestyle='None',
                          markersize=5, label='Bad Predict')
blue_dot = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None',
                          markersize=5, label='Good Predict')

fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=6,figsize=(5,30), sharex=True)
ax1.scatter(modelled_data.age, modelled_data.platelets, c=modelled_data.false, cmap="coolwarm")
ax1.legend(handles=[blue_dot, red_dot])
ax1.set(ylabel="Platelets")
ax2.scatter(modelled_data.age, modelled_data.creatinine_phosphokinase, c=modelled_data.false, cmap="coolwarm")
ax2.set(ylabel="Creatinine Phosphokinase")
ax3.scatter(modelled_data.age, modelled_data.ejection_fraction, c=modelled_data.false, cmap="coolwarm")
ax3.set(ylabel="Ejection Fraction")
ax4.scatter(modelled_data.age, modelled_data.serum_creatinine, c=modelled_data.false, cmap="coolwarm")
ax4.set(ylabel="Serum Creatinine")
ax5.scatter(modelled_data.age, modelled_data.serum_sodium, c=modelled_data.false, cmap="coolwarm")
ax5.set(ylabel="Serum Sodium")
ax6.scatter(modelled_data.age, modelled_data.time, c=modelled_data.false, cmap="coolwarm")
ax6.set(ylabel="Time");


# In[174]:


y_proba = clf_best.predict_proba(scaler.transform(X_test1))
modelled_data["predicted0"] = y_proba[:,0]
modelled_data["predicted1"] = y_proba[:,1]
# Assessing the quality of prediction: if any of predictions have a probability > 80%, we will treat is as a good one
modelled_data["pred_qual"] = np.where(np.logical_or(modelled_data["predicted0"] >0.8, modelled_data["predicted1"] >0.8), 1, 0)


# In[175]:


modelled_data["pred_qual"].sum()


# In[176]:


# And plot it again
fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=6,figsize=(5,30), sharex=True)
ax1.scatter(modelled_data.age, modelled_data.platelets, c=modelled_data.pred_qual, cmap="coolwarm")
ax1.legend(handles=[blue_dot, red_dot])
ax1.set(ylabel="Platelets")
ax2.scatter(modelled_data.age, modelled_data.creatinine_phosphokinase, c=modelled_data.pred_qual, cmap="coolwarm")
ax2.set(ylabel="Creatinine Phosphokinase")
ax3.scatter(modelled_data.age, modelled_data.ejection_fraction, c=modelled_data.pred_qual, cmap="coolwarm")
ax3.set(ylabel="Ejection Fraction")
ax4.scatter(modelled_data.age, modelled_data.serum_creatinine, c=modelled_data.pred_qual, cmap="coolwarm")
ax4.set(ylabel="Serum Creatinine")
ax5.scatter(modelled_data.age, modelled_data.serum_sodium, c=modelled_data.pred_qual, cmap="coolwarm")
ax5.set(ylabel="Serum Sodium")
ax6.scatter(modelled_data.age, modelled_data.time, c=modelled_data.pred_qual, cmap="coolwarm")
ax6.set(ylabel="Time");


# In[180]:


data_death = Heart_fail1[Heart_fail1.DEATH_EVENT==1]
data_nodeath = Heart_fail1[Heart_fail1.DEATH_EVENT==0].sample(n=100)
data_balanced = pd.concat([data_death, data_nodeath], axis=0, ignore_index=True)


# In[181]:


# Let's shuffle the balanced dataset
data_balanced_s = data_balanced.sample(frac=1)
data_balanced_s.reset_index(inplace=True, drop=True)
data_balanced_s.head()


# In[183]:


X = data_balanced_s.drop("DEATH_EVENT", axis=1)
y = data_balanced_s.DEATH_EVENT
np.random.seed(42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y, test_size=0.2)
clf_best.fit(scaler.transform(X_train2), y_train2)
clf_best.score(scaler.transform(X_test2), y_test2)


# In[184]:


# Main problem with this data set that this data is not balenced and wel catagorised. Hence data suffling may required for more precised model.


# In[185]:


cross_val_score(clf_best, scaler.transform(X), y, cv=5).mean()


# In[186]:


y_preds1 = clf_best.predict(scaler.transform(X_test1))
evaluate_preds(y_test1, y_preds1);
conf_mat = confusion_matrix(y_test1, y_preds1)
sns.heatmap(conf_mat/np.sum(conf_mat), fmt='.2%', annot=True, cbar=False,yticklabels=["No Failure", "Failure"],
    xticklabels=["Predicted No Failure", "Predicted Failure"]);


# # Accuracy is more then 91% which seems model can be considered as better model then all previous one. Precision level is also good to impliment.

# In[ ]:




