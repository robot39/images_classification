# image import
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext()
spark = SparkSession(sc)

from sparkdl import readImages
from pyspark.sql.functions import lit

img_dir = "hdfs:///flower-classify/personalities"
model_path = "hdfs:///flower-classify/persons.model"

#Read images and Create training & test DataFrames for transfer learning
jobs_df = readImages(img_dir + "/jobs").withColumn("label", lit(1))
zuckerberg_df = readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = jobs_train.unionAll(zuckerberg_train)

#dataframe for testing the classification model
test_df = jobs_test.unionAll(zuckerberg_test)

# model creation

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)
p_model.save(model_path)

# model tester

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
p_model_loaded = LogisticRegressionModel.load(model_path)
df = p_model_loaded.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
#predictions.select("filePath", "Prediction").show(truncate=False)
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
