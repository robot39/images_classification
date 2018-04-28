# image import

from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext()
spark = SparkSession(sc)

from sparkdl import readImages
from pyspark.sql.functions import lit

img_dir = "hdfs:///flower-classify/personalities"
#model_path = "hdfs:///flower-classify/models/persons.h5"
model_path = "/home/hduser/persons.h5"
model_hdfs_path = "hdfs:///flower-classify/models/persons.h5"

#Read images and Create training & test DataFrames for transfer learning
jobs_df = readImages(img_dir + "/jobs").withColumn("label", lit(1))
zuckerberg_df = readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = jobs_train.unionAll(zuckerberg_train)

#dataframe for testing the classification model
test_df = jobs_test.unionAll(zuckerberg_test)
test_df.show()

# model creation

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import readImages, TFImageTransformer
import sparkdl.graph.utils as tfx  # strip_and_freeze_until was moved from sparkdl.transformers to sparkdl.graph.utils in 0.2.0
from sparkdl.transformers import utils
import tensorflow as tf

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    image_arr = utils.imageInputPlaceholder()
    resized_images = tf.image.resize_images(image_arr, (299, 299))
    # the following step is not necessary for this graph, but can be for graphs with variables, etc
    frozen_graph = tfx.strip_and_freeze_until([resized_images], graph, sess, return_graph=True)

transformer = TFImageTransformer(inputCol="image", outputCol="features", graph=frozen_graph,
                                 inputTensor=image_arr, outputTensor=resized_images)

processed_image_df = transformer.transform(test_df)
processed_image_df.show()
#model.save(model_path)
#with open(model_path, 'rb') as f:
#	f_content = f.read()
#tf.gfile.FastGFile(model_hdfs_path, 'wb').write(f_content)
print("-------------CAN TRANSFORM--------------")

lr = LogisticRegression(maxIter=25, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[transformer, lr])
p_model = p.fit(train_df)
print("-----------CAN TRAIN-----------------")

# model tester

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = p_model.transform(test_df)
predictions.show()

predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
#predictions.select("filePath", "Prediction").show(truncate=False)
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
