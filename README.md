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

# Using PyTorch Cosine Similarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
door_sim = cos(npzfile["door"][0], npzfile["door"][1])
face_sim = cos(npzfile["face"][0], npzfile["face"][0])
stairs_sim = cos(npzfile["stairs"][0], npzfile["stairs"][0])
print('\nCosine similarity DOOR FACE: {0}\n'.format(door_sim ))
print('\nCosine similarity DOOR STAIRS: {0}\n'.format(face_sim))
print('\nCosine similarity FACE STAIRS: {0}\n'.format(stairs_sim ))


door_face_sim = cos(npzfile["door"][0].unsqueeze(0), npzfile["face"][0].unsqueeze(0))
door_stairs_sim = cos(npzfile["door"][0].unsqueeze(0), npzfile["stairs"][0].unsqueeze(0))
face_stairs_sim = cos(npzfile["face"][0].unsqueeze(0), npzfile["stairs"][0].unsqueeze(0))
print('\nCosine similarity DOOR FACE: {0}\n'.format(door_face_sim ))
print('\nCosine similarity DOOR STAIRS: {0}\n'.format(door_stairs_sim))
print('\nCosine similarity FACE STAIRS: {0}\n'.format(face_stairs_sim ))
