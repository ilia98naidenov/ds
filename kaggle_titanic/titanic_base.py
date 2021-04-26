import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt

X_full = pd.read_csv('train.csv')
X_full = X_full.dropna(subset=['Survived'], axis=0)
# print(len(X_full))

X_full = X_full.dropna(subset=['Age', 'Embarked'], axis=0)
y = X_full['Survived']
X_full = X_full.drop(['Survived'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.7, test_size=0.3, random_state=1)

X_test = pd.read_csv('test.csv')
X_test = X_test.dropna(subset=['Age', 'Fare'], axis=0)
# briefly observe what we have

numeric_cols = [cname for cname in X_full.columns
                if X_full[cname].dtype in ['int64', 'float64']]
high_cardinal_cat_cols = [cname for cname in X_full.columns
                          if X_full[cname].dtype == 'object' and X_full[cname].nunique() > 10]
low_card_cat_cols = [cname for cname in X_full.columns
                     if X_full[cname].dtype == 'object' and X_full[cname].nunique() <= 10]

# print(numeric_cols)
# print(high_cardinal_cat_cols)
# print(low_card_cat_cols)

my_cols = low_card_cat_cols + numeric_cols
X_train = X_train[my_cols]
X_valid = X_valid[my_cols]

# print(X_full.isnull().any())
# print(X_test.isnull().any())

# print(X_test.Fare.isnull().sum() / len(X_test))
# print(X_test.Age.isnull().sum() / len(X_test))

# ONE-HOT encoding

oh_encoder = OneHotEncoder(sparse=False)

to_oh = ['Embarked', 'Pclass']
oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[to_oh]))
oh_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[to_oh]))
oh_cols_test = pd.DataFrame(oh_encoder.transform(X_test[to_oh]))

oh_cols_train.index = X_train.index
oh_cols_valid.index = X_valid.index
oh_cols_test.index = X_test.index

X_train = X_train.drop(to_oh, axis=1)
X_valid = X_valid.drop(to_oh, axis=1)
X_test = X_test.drop(to_oh, axis=1)

X_train = pd.concat([X_train, oh_cols_train], axis=1)
X_valid = pd.concat([X_valid, oh_cols_valid], axis=1)
X_test = pd.concat([X_test, oh_cols_test], axis=1)

# LABEL encoding

label_encoder = LabelEncoder()
to_label = ['Sex']

for col in to_label:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_valid[col] = label_encoder.transform(X_valid[col])
    X_test[col] = label_encoder.transform(X_test[col])

# looking for correlating features (no more)

corr_coef = X_train.corr()
cor_field = []
for i in corr_coef:
    for j in corr_coef.index[corr_coef[i] > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            print("%s --> %s: r^2 = %f" % (i, j, corr_coef[i][corr_coef.index == j].values[0]))

# print(corr_coef)

# Max_Min scaling

min_max_scaler = MinMaxScaler()
to_pop = ['Pclass', 'PassengerId']
for i in to_pop:
    numeric_cols.pop(numeric_cols.index(i))
# print(numeric_cols)

X_train[numeric_cols] = min_max_scaler.fit_transform(X_train[numeric_cols])
X_valid[numeric_cols] = min_max_scaler.fit_transform(X_valid[numeric_cols])
X_test[numeric_cols] = min_max_scaler.fit_transform(X_test[numeric_cols])

print(y_valid.nunique())

# for col in X_train.columns:
#     if X_train[col].dtype == 'object':
#         print(col)


# take a look on processed data
# X_train.to_csv('processed_data.csv')

# modeling
svm = SGDClassifier(max_iter=100, loss='log')
rand_forest = RandomForestClassifier(n_estimators=200, max_depth=5)
grad_boost = GradientBoostingClassifier(n_estimators=50, max_depth=5)
xg_boost = XGBClassifier(n_estimators=50, max_depth=5)

models = [svm, rand_forest, grad_boost, xg_boost]

print('Training score:')
for model in models:
    model.fit(X_train, y_train)
    pred_train = model.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, pred_train)
    print('auc = ', score_train)

print('Validation score:')
plt.figure()
for model in models:
    pred_valid = model.predict_proba(X_valid)[:, 1]
    score_valid = roc_auc_score(y_valid, pred_valid)
    print('auc = ', score_valid)
    fpr, tpr, thresholds = roc_curve(y_true=y_valid, y_score=pred_valid)
    md = str(model)
    md = md[:md.find('(')]
    plt.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, score_valid))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()






