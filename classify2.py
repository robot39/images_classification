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
from sparkdl import DeepImageFeaturizer
from keras.applications import InceptionV3
import tensorflow as tf

model = InceptionV3(weights="imagenet")
model.save(model_path)
#with open(model_path, 'rb') as f:
#	f_content = f.read()
#tf.gfile.FastGFile(model_hdfs_path, 'wb').write(f_content)
print("-------------OK--------------")

# model tester

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from sparkdl import KerasImageFileTransformer

def loadAndPreprocessKerasInceptionV3(uri):
    # this is a typical way to load and prep images in keras
    image = img_to_array(load_img(uri, target_size=(299, 299)))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

transformer = KerasImageFileTransformer(inputCol="image", outputCol="predictions",
                                        modelFile=model_path,
                                        imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")

print(type(transformer))
final_df = transformer.transform(test_df)
final_df.show()
