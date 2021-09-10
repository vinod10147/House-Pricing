
import pyspark
from pyspark import RDD
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType
from scipy import stats
from scipy.stats import norm, skew 

from pyspark.ml.regression import RandomForestRegressionModel




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

spark_session = SparkSession.builder.master("local[2]").appName("HousingRegression").getOrCreate()

spark_context = spark_session.sparkContext

ssc = StreamingContext(spark_context,5)

message =KafkaUtils.createDirectStream(ssc,["registereduser"], {"metadata.broker.list":"localhost:9092"})

lines = message.map(lambda x: predict(dict(x[1])))

lines.pprint()

ssc.start()
ssc.awaitTermination()




spark_sql_context = SQLContext(spark_context)


def initializeModel():
	global rf_model
	rf_model=RandomForestRegressionModel.load('model_new.txt')

initializeModel()

def predict(pd_test):
	

	TEST_INPUT = './input/test.csv'


	test_df = spark_session.createDataFrame(pd_test)

	# As PySpark DFs can be finicky, sometimes your have to explicitly cast certain data types to columns

	# test_df = test_df.withColumn("BsmtFinSF1", test_df["BsmtFinSF1"].cast(IntegerType()))
	# test_df = test_df.withColumn("BsmtFinSF2", test_df["BsmtFinSF2"].cast(IntegerType()))
	# test_df = test_df.withColumn("BsmtUnfSF", test_df["BsmtUnfSF"].cast(IntegerType()))
	# test_df = test_df.withColumn("TotalBsmtSF", test_df["TotalBsmtSF"].cast(IntegerType()))
	# test_df = test_df.withColumn("BsmtFullBath", test_df["BsmtFullBath"].cast(IntegerType()))
	# test_df = test_df.withColumn("BsmtHalfBath", test_df["BsmtHalfBath"].cast(IntegerType()))
	# test_df = test_df.withColumn("GarageCars", test_df["GarageCars"].cast(IntegerType()))
	# test_df = test_df.withColumn("GarageArea", test_df["GarageArea"].cast(IntegerType()))

	# # In[37]:


	test_string_columns = []

	for col, dtype in test_df.dtypes:
	    if dtype == 'string':
	        test_string_columns.append(col)



	indexers2 = [StringIndexer(inputCol=column, outputCol=column+'_index', handleInvalid='keep').fit(test_df) for column in test_string_columns]

	pipeline2 = Pipeline(stages=indexers2)
	test_indexed = pipeline2.fit(test_df).transform(test_df)



	print(len(test_indexed.columns))


	def get_dtype(df,colname):
	    return [dtype for name, dtype in df.dtypes if name == colname][0]

	        
	num_cols_test = []
	for col in test_indexed.columns:
	    if get_dtype(test_indexed,col) != 'string':
	        num_cols_test.append(str(col))

	test_indexed = test_indexed.select(num_cols_test)


	
	print(len(test_indexed.columns))


	vectorAssembler2 = VectorAssembler(inputCols = test_indexed.columns, outputCol = 'features').setHandleInvalid("keep")

	test_vector = vectorAssembler2.transform(test_indexed)	

	test_vector = test_vector.withColumn("SalePrice", lit(0))



	

	rf_predictions2 = rf_model.transform(test_vector)
	#rf_predictions2.printSchema()
	pred = rf_predictions2.select("Id","prediction")
	pred = pred.withColumnRenamed("prediction","SalePrice")

	from pyspark.sql.types import FloatType, IntegerType

	#pred.printSchema()
	pred = pred.withColumn("Id", pred["Id"].cast(IntegerType()))
	pred = pred.withColumn("SalePrice", pred["SalePrice"].cast(FloatType()))
	print(pred.head(10))


d={'Id': 1461, 'MSSubClass': 20, 'MSZoning': 'RH', 'LotFrontage': 80.0, 'LotArea': 11622, 'Street': 'Pave', 'Alley': 'NoData', 'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Feedr', 'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 5, 'OverallCond': 6, 'YearBuilt': 1961, 'YearRemodAdd': 1961, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'MasVnrArea': 0.0, 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'CBlock', 'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'Rec', 'BsmtFinSF1': 468.0, 'BsmtFinType2': 'LwQ', 'BsmtFinSF2': 144.0, 'BsmtUnfSF': 270.0, 'TotalBsmtSF': 882.0, 'Heating': 'GasA', 'HeatingQC': 'TA', 'CentralAir': 'Y', 'Electrical': 'SBrkr', '1stFlrSF': 896, '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 896, 'BsmtFullBath': 0.0, 'BsmtHalfBath': 0.0, 'FullBath': 1, 'HalfBath': 0, 'BedroomAbvGr': 2, 'KitchenAbvGr': 1, 'KitchenQual': 'TA', 'TotRmsAbvGrd': 5, 'Functional': 'Typ', 'Fireplaces': 0, 'FireplaceQu': 'NoData', 'GarageType': 'Attchd', 'GarageYrBlt': 1961.0, 'GarageFinish': 'Unf', 'GarageCars': 1.0, 'GarageArea': 730.0, 'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y', 'WoodDeckSF': 140, 'OpenPorchSF': 0, 'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 120, 'PoolArea': 0, 'PoolQC': 'NoData', 'Fence': 'MnPrv', 'MiscFeature': 'NoData', 'MiscVal': 0, 'MoSold': 6, 'YrSold': 2010, 'SaleType': 'WD', 'SaleCondition': 'Normal'}
predict(pd.DataFrame([d]))



