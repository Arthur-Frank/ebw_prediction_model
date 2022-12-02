# Датасет используемый для обучения и проверки модели имеет 72 строки.
# Из-за минимального разброса данных SMOTE приводит к ументшению точности модели (скорее всего из-за эффекта overfitting)
# В связи с этим модель для использования в обучения и использования в приложении будет выбираться из двух регрессионных моделей:
#   - Decision Tree
#   - Random Forest
# Также будет изучена возможность улучшения результата при использовании стандартизации и нормализации
# В данном проекте осознано не используется принцип DRY для наглядности учебног проекта.

# Импорт библиотек
# Библиотеки для работы с датасетами и массивами
import numpy as np
import pandas as pd

# Библиотеки подготовки данных
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Классы Регрессоров "Дерево решений" и "Случайный лес"
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Метрика попадания предсказания модели.
from sklearn.metrics import r2_score

# библиотека создания дампа модели для использования в приложении
import pickle

# Чтение датасета из CSV файла. Необходимо построить предсказания для послдених двух характеристик Depth Width,
# на основе данных перывх 4х столбцов IW	IF	VW	FP
dataset = pd.read_csv('ebw_data.csv')
print(dataset.describe())

"""
вывод dataset.describe():

IW	IF	VW	FP	Depth	Width
count	72.000000	72.000000	72.000000	72.000000	72.000000	72.000000
mean	45.666667	141.333333	8.638889	78.333333	1.195556	1.970417
std	1.678363	5.145763	2.061078	21.493530	0.225081	0.279040
min	43.000000	131.000000	4.500000	50.000000	0.800000	1.680000
25%	44.000000	139.000000	8.000000	60.000000	1.080000	1.760000
50%	45.500000	141.000000	9.000000	80.000000	1.200000	1.840000
75%	47.000000	146.000000	10.000000	80.000000	1.290000	2.050000
max	49.000000	150.000000	12.000000	125.000000	1.760000	2.600000
"""

dataset.hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2))

# X - признаки, первые четыре столбца (все кроме последних 2х столбцов)
X = dataset.iloc[:, :-2].values

# y  - то, что нужно предсказать (2 последних столбца)
y = dataset.iloc[:, -2:].values

# Разделение на обучающую и проверочную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Обучение решающих деревьев
regressor_dto = DecisionTreeRegressor(random_state = 0)
regressor_dto.fit(X_train, y_train)

# Предсказание на основе проверочной выборки
y_pred = regressor_dto.predict(X_test)

# Проверка точности предсказания путем сравнения предсказанных выше и взятых из датасета данных
dec_tree_ord = (r2_score(y_test, y_pred))

# Обучение, предсказание и проверка точности для модели Случайного леса
regressor_rfo = RandomForestRegressor(n_estimators = 51, random_state = 0)
regressor_rfo.fit(X_train, y_train)

y_pred = regressor_rfo.predict(X_test)
rand_for_ord = (r2_score(y_test, y_pred))

#
# Нормализация датасета
#
minmax_scaleer = MinMaxScaler()

# Прогон моделей после нормализации
X_norm = minmax_scaleer.fit_transform(X)
y_norm = minmax_scaleer.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size = 0.2, random_state = 0)

regressor_dtn = DecisionTreeRegressor(random_state = 0)
regressor_dtn.fit(X_train, y_train)

y_pred = regressor_dtn.predict(X_test)
dec_tree_norm = (r2_score(y_test, y_pred))

regressor_rfn= RandomForestRegressor(n_estimators = 51, random_state = 0)
regressor_rfn.fit(X_train, y_train)

y_pred = regressor_rfn.predict(X_test)
rand_for_norm = (r2_score(y_test, y_pred))
print(X_test)
print(y_pred)

#
# Стандартизация и прогон моделей после стандартизации
#

sc_X = StandardScaler()
sc_y = StandardScaler()
X_std = sc_X.fit_transform(X)
y_std = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size = 0.2, random_state = 0)

regressor_dts= DecisionTreeRegressor(random_state = 0)
regressor_dts.fit(X_train, y_train)

y_pred = regressor_dts.predict(X_test)
dec_tree_std= (r2_score(y_test, y_pred))

regressor_rfs = RandomForestRegressor(n_estimators = 51, random_state = 0)
regressor_rfs.fit(X_train, y_train)

y_pred = regressor_rfs.predict(X_test)
rand_for_std = (r2_score(y_test, y_pred))

#
# Прогон модели после стандартизации и нормализации
#

#применяем стандартизацию на уже существующие нормализованные данные

X_norm_std = sc_X.fit_transform(X_norm)
y_norm_std = sc_y.fit_transform(y_norm)

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size = 0.2, random_state = 0)

regressor_dtsn = DecisionTreeRegressor(random_state = 0)
regressor_dtsn.fit(X_train, y_train)

y_pred = regressor_dtsn.predict(X_test)
dec_tree_norm_std= (r2_score(y_test, y_pred))

regressor_rfsn = RandomForestRegressor(n_estimators = 51, random_state = 0)
regressor_rfsn.fit(X_train, y_train)

y_pred = regressor_rfsn.predict(X_test)
rand_for_norm_std = (r2_score(y_test, y_pred))


#сводим в таблицу и выводим на экран точности моделей всех использованных вариантов
result_data = {'model':['DecisionTree','RandomForestRegressor'],
               'ordinary':[dec_tree_ord, rand_for_ord],
               'normalized':[dec_tree_norm, rand_for_norm],
               'standardized':[dec_tree_std, rand_for_std],
               'normalized and standardized':[dec_tree_norm_std, rand_for_norm_std]}

df = pd.DataFrame(result_data)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

"""
вывод таблицы:
             model       ordinary  normalized  standardized  normalized and standardized  
          DecisionTree  0.859833    0.859833      0.859833   0.859833
 RandomForestRegressor  0.874448    0.875617      0.874239   0.874239

Модель для дальнейшего использования в приложении - Случайный лес. Манипуляция с данными не смога повысить точность,
однако может привести к существенному изменению качества предсказания в случае изменения размера выборки.
Для упрощения выбирается случай без стандартизации и нормализации.
"""

# Создание дампа модели для переноса в приложение
with open('ebw_model_predicter.pkl', 'wb') as file:
    pickle.dump(regressor_rfo, file)



