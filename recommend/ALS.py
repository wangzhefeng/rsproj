# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


from collections import defaultdict
from ..utils.matrix import Matrix
from random import random


class ALS(object):

	def __init__(self):
		"""
		初始化存储:
		用户ID, 物品ID, 用户ID与用户矩阵列号的对应关系,物品ID与物品矩阵列号的对应关系
		用户已经看到那些物品
		评分矩阵的Shape
		RMSE
		"""
		self.user_ids = None
		self.item_ids = None
		self.user_ids_dict = None
		self.item_ids_dict = None
		self.user_matrix = None
		self.item_matrix = None
		self.shape = None
		self.rmse = None

	def _process_data(self, X):
		"""
		数据预处理
		:param X:
		:return:
		"""
		self.user_ids = tuple((set(map(lambda x: x[0], X))))
		self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))
		self.item_ids = tuple((set(map(lambda x: x[1], X))))
		self.item_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.item_ids)))
		self.shape = (len(self.user_ids), len(self.item_ids))

		ratings = defaultdict(lambda: defaultdict(int))
		ratings_T = defaultdict(lambda: defaultdict(int))

		for row in X:
			user_id, item_id, rating = row
			ratings[user_id][item_id] = rating
			ratings_T[item_id][user_id] = rating

		err_msg = "Length of user_ids %d and rating %d not match!" % (len(self.user_ids), len(ratings))
		assert len(self.user_ids) == len(ratings), err_msg

		return ratings, ratings_T


	def _users_mul_ratings(self, users, ratings_T):
		"""
		用户矩阵乘以评分矩阵
		:param users:
		:param ratings_T:
		:return:
		"""
		def f(users_row, item_id):
			user_ids = iter(ratings_T[item_id].keys())
			scores = iter(ratings_T[item_id].values())
			col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
			_users_row = map(lambda x: users_row[x], col_nos)

			return sum(a * b for a, b in zip(_users_row, scores))

		ret = [[f(users_row, item_id) for item_id in self.item_ids] for users_row in users.data]

		return Matrix(ret)


	def _items_mul_ratings(self, items, ratings):
		"""
		物品矩阵乘以评分矩阵
		:param imtes:
		:param ratings:
		:return:
		"""
		def f(items_row, user_id):
			item_ids = iter(ratings[user_id].keys())
			scores = iter(ratings[user_id].values())
			col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
			_items_row = map(lambda x: items_row[x], col_nos)

			return sum(a * b for a, b in zip(_items_row, scores))

		ret = [[f(items_row, user_id) for user_id in self.user_ids] for items_row in items.data]

		return Matrix(ret)


	def _gen_random_matrix(self, n_rows, n_columns):
		"""
		生成随机矩阵
		:param n_rows:
		:param n_columns:
		:return:
		"""
		data = [[random() for _ in range(n_columns)] for _ in range(n_rows)]
		return Matrix(data)


	def _get_rmse(self, ratings):
		"""
		计算RMSE
		:param ratings:
		:return:
		"""
		m, n = self.shape
		mse = 0.0
		n_elements = sum(map(len, ratings.values()))
		for i in range(m):
			for j in range(n):
				user_id = self.user_ids[i]
				item_id = self.item_ids[j]
				rating = ratings[user_id][item_id]
				if rating > 0:
					user_row = self.user_matrix.col(i).transpose
					item_col = self.item_matrix.col(j)
					rating_hat = user_row.mat_mul(item_col).data[0][0]
					square_error = (rating - rating_hat) ** 2
					mse += square_error / n_elements
		return mse ** 0.5

	def fit(self, X, k, max_iter = 10):
		"""
		1.数据预处理
		2.变量合法性检查
		3.生成随机矩阵U
		4.交替计算矩阵U和矩阵I, 并打印RMSE, 直到迭代次数达到max_iter
		5.保存最终的RMSE
		:param X:
		:param k:
		:param max_iter: 最大迭代次数
		:return:
		"""
		ratings, ratings_T = self._process_data(X)
		self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
		m, n = self.shape
		error_msg = "Parameter k must be less than the rank of original matrix"
		assert k < min(m, n), error_msg

		self.user_matrix = self._gen_random_matrix(k, m)

		for i in range(max_iter):
			if i % 2:
				items = self.item_matrix
				self.user_matrix = self._items_mul_ratings(
					items.mat_mul(items.transpose).inverse.mat_mul(items), ratings
				)
			else:
				users = self.user_matrix
				self.item_matrix = self._users_mul_ratings(
					users.mat_mul(users.transpose).inverse.mat_mul(users), ratings_T
				)
			rmse = self._get_rmse(ratings)
			print("Iterations: %d. RMSE: %.6f" % (i + 1, rmse))

		self.rmse = rmse

	def _predict(self, user_id, n_items):
		"""
		预测一个用户感兴趣的内容, 剔除用户已经看过的内容
		然后按感兴趣分值排序, 取出前n_items个内容
		:param user_id:
		:param n_items:
		:return:
		"""
		users_col = self.user_matrix.col(self.user_ids_dict[user_id])
		users_col = users_col.transpose

		items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
		items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
		viewed_items = self.user_items[user_id]
		items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)

		return sorted(items_scores, key = lambda x: x[1], reverse = True)[:n_items]

	def predict(self, user_ids, n_items = 10):
		"""
		循环调用_predict, 预测多个用户感兴趣的内容
		:param user_ids:
		:param n_items:
		:return:
		"""
		return [self._predict(user_id, n_items) for user_id in user_ids]


@run_time
def main():
	print("Testing the accuracy of ALS...")
	X = load_movie_ratings()
	model = ALS()
	model.fit(X, k = 3, max_iter = 5)
	print()

	print("Showing the predictions of users...")

	user_ids = range(1, 5)
	predictions = model.predict(user_ids, n_items = 2)
	for user_id, prediction in zip(user_ids, predictions):
		_prediction = [format_prediction(item_id, score) for item_id, score in prediction]
		print("User id: %d recommedation: %s" % (user_id, _prediction))

