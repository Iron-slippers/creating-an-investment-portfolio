import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import statsmodels as smt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso


def calculate_income(paths_cost, paths_indicator):
	result_bank = {'Name': [],
				   'Dangerous': [],
				   'Predict_income_aprocs': [],
				   'Predict_income_arima': [],
				   'Predict_indicator': [],
				   'Predict_income': [],
				   'Real_income': []}

	for url_cost, url_indicator in tqdm(zip(paths_cost, paths_indicator)):
		answer_model = work_with_stock(url_cost, url_indicator)
		name = url_cost.split('/')[2].split('_')[0]

		for index, key in enumerate(result_bank.keys()):

			if key == 'Name':
				result_bank[key].append(name)
				continue

			result_bank[key].append(answer_model[index - 1])

	result = pd.DataFrame(result_bank)

	answer = []
	mean = round(np.mean(result['Real_income']) * 100, 1)
	answer.append(f'Обычная прибыль: {mean}%')

	for risk in [0.3, 0.5, 0.7]:
		answer.append(check_income_of_risk(risk, result))

	return result, answer


# The function is used to return profits
def check_income_of_risk(risk_koefficient, df):
	result = df[(df['Dangerous'] <= risk_koefficient) & (df['Predict_income'] > 0.05)]
	predict_income_sum = sum(result['Predict_income'])
	income = 0

	for index, row in result.iterrows():
		income += (row['Predict_income'] / predict_income_sum) * row['Real_income']

	return f'Риск = {risk_koefficient}. Прибыль = {round(income * 100, 1)}%'


# Calculation of risk values, future value
def work_with_stock(path_cost, path_indicator):
    data = preproccesing_cost(path_cost)
    result = common_indicator(path_cost, path_indicator)

    size, delta = data.shape[0], 50
    seasonal_year, trend_year, resid_year = decompose(data.iloc[:size - delta, 1], 52)

    x = [(1 / size) * index for index in range(size - delta)]
    popt, pcov = curve_fit(calculate, x, data['target'][:-delta])

    aprocs, for_sigma = [], []
    index_start = 0

    for index, value in enumerate([(1 / size) * index for index in range(size)]):

        if index >= size - delta and not index_start:
            index_start = index

        res = calculate(value, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

        if index >= size - delta:
            res += seasonal_year[2 * index_start - index - 1]
            for_sigma.append(seasonal_year[2 * index_start - index - 1])

        aprocs.append(res)

    model = creat_model_sarima(data.iloc[:-delta, 1], trend=[0, 1], order=(2, 1, 52), seasonal_order=(0, 0, 0, 0))

    dangerous = max(seasonal_year + resid_year)
    predict_income_aprocs = round(aprocs[-1] - data.iloc[-50, 1], 3)
    predict_income_arima = round(model.predict(size, size).values[0] - data.iloc[-50, 1], 3)
    predict_income = round((predict_income_aprocs + 2 * predict_income_arima + 3 * result[0]) / 7, 3)
    real_income = round(data.iloc[-1, 1] - data.iloc[-50, 1], 3)

    return dangerous, predict_income_aprocs, predict_income_arima, np.mean(result), predict_income, real_income


# Predicting the future value of shares by reporting
def common_indicator(url_cost, url_indicator):
	data_cost = preproccesing_cost(url_cost)
	data_indicator = preproccesing_indicator(url_indicator, data_cost)

	data_targets = get_predict_median(data_indicator, data_cost)
	size = int(data_indicator.shape[0] - 1)
	X_train, X_test, y_train, y_test = data_indicator[:size], data_indicator[size:], data_targets[:size], data_targets[
																										  size:]

	result = []
	for column in data_targets.columns:
		model_ready = creat_model_lasso(X_train.drop('date', axis=1), y_train[column])
		predict = model_ready.predict(X_test.drop('date', axis=1))
		result.append(predict[-1])

	return result


def preproccesing_indicator(path, costs):
	data = pd.read_csv(path, delimiter=',')

	columns = list(data['Unnamed: 0'])
	data = data.T
	data.columns = np.array(columns)
	data.drop('Unnamed: 0', inplace=True)
	if 'Цена акции ап, руб' in data.columns:
		data.drop('Цена акции ап, руб', axis=1, inplace=True)

	data['date'] = [generate_datetime(float(value)) for value in data.index]
	data.reset_index(drop=True, inplace=True)

	data = data[(data['date'] >= costs.iloc[0, 0]) & (data['date'] < costs.iloc[-1, 0])].copy()
	data.reset_index(drop=True, inplace=True)

	data.dropna(axis=1, inplace=True)
	return data


def preproccesing_cost(path):
	data = pd.read_csv(path, delimiter=';')

	data.drop(['<TIME>', '<HIGH>', '<LOW>', '<CLOSE>'], axis=1, inplace=True)
	data.columns = ['date', 'target', 'volume']
	data['date'] = data.apply(norm_view_date, axis=1)

	scaler = MinMaxScaler()
	scaler.fit(np.array(data['target']).reshape(-1, 1))
	data['target'] = scaler.transform(np.array(data['target']).reshape(-1, 1))

	return data

# Calculating the target for a linear model
def get_predict_median(data_indicator, data_cost):
	date_list = data_indicator['date']
	index_date = 0
	median, min_targets, max_targets, costs = [], [], [], []

	for index, row in data_cost.iterrows():

		if row['date'] < date_list[index_date]:
			continue

		if index_date + 1 < len(date_list) and row['date'] >= date_list[index_date + 1]:
			median.append(np.median(costs))
			min_targets.append(min(costs))
			max_targets.append(max(costs))
			costs.clear()
			index_date += 1

		costs.append(row['target'])

	median.append(np.median(costs))
	max_targets.append(max(costs))
	min_targets.append(min(costs))

	result = {
		'median': median,
		'max_targets': max_targets,
		'min_targets': min_targets
	}

	return pd.DataFrame(result)


def creat_model_lasso(features, target):
	model = Lasso()
	model.fit(features, target)
	return model


def norm_view_date(row):
    date = str(row['date'])
    return datetime(int(date[:4]), int(date[4:6]), int(date[6:8]))


def generate_datetime(value):
	year = int(value)
	month = int(12 * (value - year) + 1)
	return datetime(year, month, 1)


def decompose(time_series, period=365):
    result = STL(time_series, period=period).fit()
    return result.seasonal, result.trend, result.resid


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calculate(x, b, c, d, e, f, g, h, i):
    return sigmoid(b * np.sin(x) + c * np.cos(x) + d * np.sin(13 * x) + e * np.cos(13 * x) + f * np.sin(4.5 * x) + g * np.cos(4.5 * x) + h * np.sin(52 * x) + i * np.cos(52 * x))


def creat_model_sarima(y, trend=[0, 0], order=(2, 0, 52), seasonal_order=(0, 0, 0, 0)):
    model = SARIMAX(y, trend=trend, order=order, seasonal_order=seasonal_order)
    return model.fit()


bank, answer = calculate_income(paths_cost, paths_indicator)
