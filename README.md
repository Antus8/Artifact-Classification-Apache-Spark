# Artifact-Classification-Apache-Spark
Artifact classification using Apache Spark and Pytorch.
Recognized classes are: doors, faces, stairs and pedestrians.

# NOTES: 
Solo pandas riesce a leggere bene i numpy arrays ma poi ci sono problemi con il modello di ML
Quindi convertiamo i numpy arrays in liste, così spark riesce a inferire lo schema
df = pd.DataFrame(dataset)
print(df)

indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
indexed_df = indexer.fit(df).transform(df)
indexed.show()

IL MODELLO NON ACCETTA ARRAY MA SOLO VECTOR TYPE, QUINDI DOBBIAMO RICONVERTIRE


# GRADIENT BOOSTED TREE

Only supports BINARY classification

gbt = GBTClassifier(labelCol="label_indexed", featuresCol="vector_features", maxIter=10)
pipeline = Pipeline(stages=[labelIndexer, gbt, labelConverter])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.show(50)


evaluator = MulticlassClassificationEvaluator(
    labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
gbt_accuracy = evaluator.evaluate(predictions)
'''
