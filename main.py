# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-24
# * Version     : 0.1.112422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
pd.set_option("max_columns", None, "max_rows", None)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data 
data = [
    ['牛奶','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
    ['莳萝','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
    ['牛奶','苹果','芸豆','鸡蛋'],
    ['牛奶','独角兽','玉米','芸豆','酸奶'],
    ['玉米','洋葱','洋葱','芸豆','冰淇淋','鸡蛋']
]

# data preprocessing(# one-hot encode)
te = TransactionEncoder()
te_array = te.fit(data).transform(data)
print(te_array)

# data frame
df = pd.DataFrame(te_array, columns = te.columns_)
print(df)

# apriori analysis
freq = apriori(df, min_support = 0.05, use_colnames = True)
print(freq.head())
print(freq.tail())

# association rules
result = association_rules(
    freq, 
    metric = "confidence", 
    min_threshold = 0.6
)
print(result.head())
print(result.tail())
print(result.columns)

result.sort_values(by = 'confidence', ascending = False)
print(result.head())
print(result.tail())








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

