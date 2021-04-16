import numpy as np
import pandas as pd
import os
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            ЭТАП ПРЕДОБРАБОТКИ И АНАЛИЗА ДАННЫХ
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

table_name = 'data.csv'
table = pd.read_csv(table_name, sep=';')

# print(table.head(15))
# предпросмотр таблички

table = table.drop_duplicates()
# избавляемся от полностью повторяющихся строк
# print(len(table))
table = table.dropna()
# избавились от NaN

# print(True in table.isnull())
# проверяем, все ли поля заполнены


# num_clients = table.client_id.count()
# num_distinct_clients = table.client_id.drop_duplicates().count()
# print(num_clients)
# print(num_distinct_clients)
# проверяем, что у одного и того же клиента может быть несколько заказов. Это нормально.
# id клиента информативен, потому что если он отдал один кредит, он вероятнее всего, отдаст и второй

# num_orders = table.order_id.count()
# num_distinct_orders = table.order_id.drop_duplicates().count()
# print(num_orders)
# print(num_distinct_orders)
# выясняем, что один order_id может соответствовать сразу нескольким заказам, даже после drop_duplicates.
# Этого, по логике, быть не должно.

orders = Counter(table.order_id)
# print(orders)
# Посмотрим, сколько вхождений в колонку имеет каждый id-шник

repeated_order_ids = []
for key in orders.keys():
    rank = orders.get(key)
    if rank > 1:
        repeated_order_ids.append(key)

# выведем повторяющиеся id-шники
# print(repeated_order_ids)

repeated = sum([table.order_id == i for i in repeated_order_ids])
repeated = np.array(repeated, dtype=bool)
# фрагмент кода ищет индексы строк, order_id которых не уникален

test = table.loc[repeated]
test.to_csv('test.csv')
# смотрим разницу между строчками с повторяющимися id
# убеждаемся, что разница незначительна и вызвана, скорее всего, опечатками
# удаляем временную csv-таблицу
os.remove('test.csv')

# удаляем строки с повторяющимися order_id, оставляя по одной из "копий"
table = table.drop_duplicates(subset=['order_id'])

# date = table.order_date
# print(min(date), max(date))
# смотрим релевантность данных.
# Интервал времени (3 месяца в 2017 году) включает в себя узкий спектр возможных экономических ситуаций,
# поэтому было принято решение не использовать переменную даты в обучающей выборке.
# (К примеру, качество предсказания на 2020 карантинный год было бы сомнительным)

# print(len(table.region[table.region == 0]))
# print(Counter(table.region))
# Смотрим, сколько записей имеют нулевой регион. Их -- треть данных.
# Остальные регионы встречаются минимум в 6 раз реже.
# убирать поле Регион не будем, потому что, скорее всего, это значит, что 0 -- родной регион
# предлагаю разделить регионы на родной и все остальные (бинарный признак)

# print(min(table.age), max(table.age))
# Смотрим максимальный и минимальный возраст кредитуемых.
# Можно отбросить значения выше 80 в силу российских реалий

# print(len(table.month_income[table.month_income == 0]))
# Смотрим количество безработных. Их -- четверть всех кредитованных.
# Посмотрим, скольким из безработных дали кредит, чтобы убедиться, что данные в порядке:

filter_unemployed = np.array(table.month_income == 0)
filter_credited = np.array(table.expert == 1)
filter_no_creds = np.array(table.closed_creds == 0)
filter_align = [filter_unemployed[i] * filter_credited[i] * filter_no_creds[i] for i in range(len(table))]
filter_align = np.array(filter_align, dtype=bool)

# print(len(table[filter_align]))
# к данным возникает вопрос, почему люди без зарплаты и без кредитной истории получают кредит.
# В дальнейшем я решил их не использовать при построении модели.

# Проанализируем просрочки в днях:
overdue_days = np.array(table.active_cred_day_overdue)
ind = table.index
# less_than_year = np.where(overdue_days < 365)
# overdue_days = overdue_days[less_than_year]
# ind = table.index[less_than_year]
# plt.plot(ind, overdue_days)

# print(np.mean(overdue_days))
# получается, что в среднем просрочка составляет чуть меньше 2х лет

# также наблюдаем пики в данных. Можно их отбросить, применив квантиль, близкий к единице
q = table.active_cred_day_overdue.quantile(0.999)
# print(q)
# Значения выше q будем отбрасывать, чтобы избавиться от резких пиков. Эти пики соответствуют
# крайне большим значениям просрочки

expert = np.array(table.expert)
# print(sum(expert == 1) / len(table))
# оценим, какой класс (0 или 1) преобладает в данных. Классы получились неравномощные.


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                      ЭТАП ФИЛЬТРАЦИИ ДАННЫХ
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


table = table.drop(table.loc[filter_align].index)
# отбрасываем кредитованных безработных без истории

table = table.drop(table.loc[table.age > 80].index)
# отбрасываем пожилых людей (для российских реалий)

table = table.drop(table.loc[table.active_cred_day_overdue > q].index)
# отбрасываем пики во времени просрочки с помощью 0.999 квантиля

table = table.drop(table.loc[table.active_cred_sum_overdue > table.active_cred_sum].index)
# отбрасываем данные, где задолженность по кредитам больше всей суммы кредитов

# print(1 - len(table) / 50000)
# в итоге отбросили около 20% данных. В принципе, я не знаю на опыте, насколько это плохо.
# Однако я думаю, что физический смысл отброшенных данных не соответствовал нашим требованиям.

# overdue_days = np.array(table.active_cred_day_overdue)
# ind = table.index
# # less_than_year = np.where(overdue_days < 365)
# # overdue_days = overdue_days[less_than_year]
# # ind = table.index[less_than_year]
# plt.plot(ind, overdue_days)
# Заново строил график просроченных дней, чтобы посмотреть, как удалились пики


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               ЭТАП ПРЕОБРАЗОВАНИЯ ПЕРЕМЕННЫХ
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


col_labels = []
columns = []

regions = np.array(table.region)
for i in range(len(regions)):
    if regions[i] != 0:
        regions[i] = 1
# преобразование регионов; превращаем регион в бинарный признак (свой/чужой)
col_labels.append('regions')
columns.append(regions)


overdue_days = np.array(table.active_cred_day_overdue)
min_days = min(overdue_days)
max_days = max(overdue_days)
overdue_days_norm = (overdue_days - min_days) / (max_days - min_days)
# Здесь я хотел сделать бинарный признак (0, если нет просрочки больше 2 месяцев, и 1, если есть).
# Однако, средняя просрочка (поскольку суммарная) составляет около 2 лет, и судя по всему, зависит от кол-ва
# взятых и просроченных кредитов. Мы не можем судить по этим данным о среднем времени просрочки для одного кредита,
# поэтому я просто отнормировал этот вектор по min/max.
col_labels.append('active_cred_day_overdue')
columns.append(overdue_days_norm)


month_income = np.array(table.month_income)
min_income = min(month_income)
max_income = max(month_income)
month_income_norm = (month_income - min_income) / (max_income - min_income)
# сосчитали среднюю з/п по всем работящим персонажам, нормировали вектор з/п по min/max
# нормировка по min/max годится для неотрицательных значений переменной
col_labels.append('month_income')
columns.append(month_income_norm)


closed_creds = np.array(table.closed_creds)
closed_cred_sum = np.array(table.closed_cred_sum)
for i in range(len(closed_cred_sum)):
    if closed_creds[i] != 0:
        closed_cred_sum[i] = closed_cred_sum[i] / closed_creds[i]
    else:
        closed_cred_sum[i] = 0
# считаем для каждого заказа сумму, которую в среднем клиент берет в кредит.
# Заодно проверяем, что если число закрытых кредитов = 0, то и сумма по всем кредитам = 0.
# Сумма неотрицательна, нормируем по min/max
max_cred = max(closed_cred_sum)
min_cred = min(closed_cred_sum)
closed_cred_sum_norm = (closed_cred_sum - min_cred) / (max_cred - min_cred)
# нормировали
col_labels.append('closed_cred_sum')
columns.append(closed_cred_sum_norm)


max_creds = max(closed_creds)
min_creds = min(closed_creds)
closed_creds_norm = (closed_creds - min_creds) / (max_creds - min_creds)
col_labels.append('closed_creds')
columns.append(closed_creds_norm)


# Дальше преобразования посложнее
# расчет кредитного потенциала
loan_cost = np.array(table.loan_cost_all)
first_days = np.array(table.first_days_quant)
potential = np.zeros(shape=loan_cost.shape)
for i in range(len(loan_cost)):
    potential[i] = month_income[i] - loan_cost[i] / first_days[i] * 30
# потенциал может быть отрицательным, нормируем по максимуму модуля
max_abs_potential = max(abs(potential))
potential_norm = potential / max_abs_potential
col_labels.append('credit_potential')
columns.append(potential_norm)
# plt.plot(table.index, potential_norm)

# расчет "надежности" клиента. Узнаем, сколько в процентах составляет доля P задолженности
# от всей суммы активных кредитов. 1 - P = надежность.
# Для некредитованных лиц воспользуемся математическим ожиданием
# "надежности", взятым с оставшейся выборки.
active_cred_sum = np.array(table.active_cred_sum)
active_cred_sum_overdue = np.array(table.active_cred_sum_overdue)
reliability = np.zeros(shape=active_cred_sum.shape)
for i in range(len(reliability)):
    if active_cred_sum[i] != 0:
        reliability[i] = 1 - active_cred_sum_overdue[i] / active_cred_sum[i]

# a = np.where(active_cred_sum < active_cred_sum_overdue)
# print(len(a[0]))
# оказывается, есть данные, где суммарная задолженность превышает суммарный активный кредит, что
# логически невозможно. Вернемся к этапу фильтрации и уберем лишние данные.
# plt.scatter(table.index, reliability)
# Теперь, как и должно быть, надежность принимает значения от 0 до 1

# подсчитаем ее мат. ожидание
k = 0
s = 0
for i in range(len(reliability)):
    if active_cred_sum[i] != 0:
        k += 1
        s += reliability[i]
mean_reliability = s / k
# print(mean_reliability)
# средняя "надежность" 0.67
inds = np.where(active_cred_sum == 0)
reliability[inds] = mean_reliability
# print(reliability[0:10])
col_labels.append('reliability')
columns.append(reliability)

# некоторые колонки просто нормировал
age = np.array(table.age)
age_norm = (age - min(age)) / (max(age) - min(age))
col_labels.append('age')
columns.append(age_norm)

active_cred_sum_norm = (active_cred_sum - min(active_cred_sum)) / (max(active_cred_sum) - min(active_cred_sum))
col_labels.append('active_cred_sum')
columns.append(active_cred_sum_norm)

gender = table.gender
col_labels.append('gender')
columns.append(gender)

max_overdue = np.array(table.active_cred_max_overdue)
max_overdue_norm = (max_overdue - min(max_overdue)) / (max(max_overdue) - min(max_overdue))
col_labels.append('max_overdue')
columns.append(max_overdue_norm)

# print(len(columns))
# смотрим, сколько признаков получилось (11)
# не вошли колонки: expert (не относ. к признакам)
#                   order_id, id (бессмысленны по большей части)
#                   cost_all (данные непонятны, и не понял, как обсчитать)
#                   first_loan (коррелирует с loan_cost)
#                   first_days (информация вшита в кред потенциал)
#                   active_cred_overdue (вшито в коэфф надежности)
#

# print(len(col_labels), col_labels)
# увидели колонки, их всего 11

data = dict(zip(col_labels, columns))
# собрали колонки в датафрейм
features_matrix = pd.DataFrame(data)
# print(features_matrix.head(20))

corr_coef = features_matrix.corr()
# смотрим корреляцию между колонками
# print(corr_coef)
# print(None in corr_coef)

cor_field = []
for i in corr_coef:
    for j in corr_coef.index[corr_coef[i] > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            # print("%s --> %s: r^2 = %f" % (i, j, corr_coef[i][corr_coef.index == j].values[0]))
# выведем столбцы, сильно коррелирующие между собой. такой всего один.

features_matrix = features_matrix.drop('month_income', axis=1)
print(len(features_matrix))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                       ЭТАП МОДЕЛИРОВАНИЯ
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


target = table.expert
train = features_matrix.values
# coder = PCA(n_components=10)
# train = coder.fit_transform(train)
# print(train.shape, features_matrix.shape)

# применение метода главных компонент для понижения размерности выборки.
# Понижение размерности -- своего рода регуляризация.

# Модели -- случайный лес, градиентный бустинг, SVM (градиентный спуск).
# Оценка -- roc, auc. Причина такой оценки -- возможность наглядно сравнить все три метода.
# также простой подсчет точности n_correct / n_all неуместен, потому что классы неравномощны.

svm = SGDClassifier(max_iter=10, loss='log')
rand_forest = RandomForestClassifier(n_estimators=200, max_depth=5)
grad_boost = GradientBoostingClassifier(n_estimators=200, max_depth=5)
models = [svm, rand_forest, grad_boost]


trn_train, trn_test, tar_train, tar_test = train_test_split(train, target, test_size=0.3, random_state=0)
# Разбили выборку на обучающую и тестовую
# print(len(trn_test), len(tar_test))

print('Training_score:')
for model in models:
    model.fit(trn_train, tar_train)
    pred_train = model.predict_proba(trn_train)[:, 1]
    score_train = roc_auc_score(tar_train, pred_train)
    print('auc = ', score_train)
# подчитываем аук для обучающей выборки

print('Testing_score:')
for model in models:
    pred_test = model.predict_proba(trn_test)[:, 1]
    score_test = roc_auc_score(tar_test, pred_test)
    print('auc = ', score_test)
    fpr, tpr, thresholds = roc_curve(y_true=tar_test, y_score=pred_test)
    md = str(model)
    md = md[:md.find('(')]
    plt.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, score_test))
# подсчитываем аук для тестовой выборки, строим roc.

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()
