#
# import tensorflow as tf
# import tensorflow_datasets as tfds
#
# datasets = tfds.list_builders()
# print(datasets)
#
# data, info = tfds.load("mnist", with_info=True)
# train_data, test_data = data['train'], data['test']
#
# print(info)

# t_1 = tf.linspace(1.0,10.0,2)
# t_2 = tf.zeros([5,6], tf.int32)
# print(t_1)
# print(t_2)
# t_1 = tf.random.normal([5,5], mean=0, stddev=1)
# t_a = tf.Variable(t_1)

#
# weights = tf.Variable(tf.random.uniform([100,100], 0,2))
# weights_2 = tf.Variable(weights.initialized_value())
#
# import tensorflow as tf
# import numpy as np
# import tensorflow_datasets as tfds
# num_items = 100
# num_list = np.arange(num_items)
# # create the dataset from numpy array
# num_list_dataset = tf.data.Dataset.from_tensor_slices(num_list)
# print(num_list)
# print(num_list_dataset)
# datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# # print(weights_2)
#
# train_dataset = datasets['train']
# train_dataset = train_dataset.batch(5).shuffle(50).take(2)
# for data in train_dataset:
# print(data)




##########33 Estimators Linear Regression ############
# import tensorflow as tf
# from tensorflow import feature_column as fc
# numeric_column = fc.numeric_column
# categorical_column = fc.categorical_column_with_vocabulary_list
#
#
# featcols = [
# numeric_column("area"),
# categorical_column("type",[
# "bungalow","apartment"])
# ]
#
# def train_input_fn():
#     features = {"area":[1000,2000,4000,1000,2000,4000],
#     "type":["bungalow","bungalow","house",
#     "apartment","apartment","apartment"]}
#     labels = [ 500 , 1000 , 1500 , 700 , 1300 , 1900 ]
#     return features, labels
#
# model = tf.estimator.LinearRegressor(featcols)
# model.train(train_input_fn, steps=200)
#
# def predict_input_fn():
#     features = {"area":[1500,1800],
#     "type":["house","apt"]}
#     return features
# predictions = model.predict(predict_input_fn)
#
#
# print(next(predictions))
# print(next(predictions))



############# Estimator with dataset from tensorflow #############
# import tensorflow as tf
# from tensorflow import feature_column as fc
# import numpy as np
# import pandas as pd
#
#
#
#
# from tensorflow.keras.datasets import boston_housing
# (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
#
# features = ['CRIM', 'ZN',
# 'INDUS','CHAS','NOX','RM','AGE',
# 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# x_train_df = pd.DataFrame(x_train, columns= features)
# x_test_df = pd.DataFrame(x_test, columns= features)
# y_train_df = pd.DataFrame(y_train, columns=['MEDV'])
# y_test_df = pd.DataFrame(y_test, columns=['MEDV'])
# x_train_df.head()
#
# feature_columns = []
# for feature_name in features:
#     feature_columns.append(fc.numeric_column(feature_name,
#     dtype=tf.float32))
#
#
# def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True,
# batch_size=32):
#     def input_function():
#         ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
#         if shuffle:
#             ds = ds.shuffle(100)
#         ds = ds.batch(batch_size).repeat(epochs)
#         return ds
#     return input_function
# train_input_fn = estimator_input_fn(x_train_df, y_train_df)
# val_input_fn = estimator_input_fn(x_test_df, y_test_df, epochs=1,
# shuffle=False)
#
# linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
# linear_est.train(train_input_fn, steps=100)
# result = linear_est.evaluate(val_input_fn)
#
# result = linear_est.predict(val_input_fn)
# for pred,exp in zip(result, y_test[:32]):
#     print("Predicted Value: ", pred['predictions'][0], "Expected:", exp)


