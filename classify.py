# image import
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext()
spark = SparkSession(sc)

from sparkdl import readImages
from pyspark.sql.functions import lit

img_dir = "hdfs:///flower-classify/flowers"

#Read images and Create training & test DataFrames for transfer learning
daisy_df = readImages(img_dir + "/daisy").withColumn("label", lit(0))
dandelion_df = readImages(img_dir + "/dandelion").withColumn("label", lit(1))
roses_df = readImages(img_dir + "/roses").withColumn("label", lit(2))
sunflowers_df = readImages(img_dir + "/sunflowers").withColumn("label", lit(3))
tulips_df = readImages(img_dir + "/tulips").withColumn("label", lit(4))

#dataframe for training a classification model
train_df = daisy_train.unionAll(dandelion_train).unionAll(roses_train).unionAll(sunflowers_train).unionAll(tulips_train)
print("TRAIN DF PREPARED.")

#dataframe for testing the classification model
test_df = daisy_test.unionAll(dandelion_test).unionAll(roses_test).unionAll(sunflowers_test).unionAll(tulips_test)

# model creation

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=5, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

# model tester

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

df = p_model.transform(test_df)
df.cache()
df.show()
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
