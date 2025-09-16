package linearregression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

object Application {
  def main(args: Array[String]): Unit ={

    //initialize spark session using all cores
    val spark = SparkSession.builder().appName("Price Prediction")
      .master("local[*]")
      .getOrCreate()

    //load data
    //make more generic/add automatic download?
    val dataPath = "src/main/resources/all-cards-20250916092739.json"
    val scryfallDataDF = spark.read.json(dataPath)
    scryfallDataDF.show()

    //clean data - create utils for this and clean more fields
    val cleanedDataDF = scryfallDataDF.select("mana_cost", "game_changer"
        , "loyalty", "power", "toughness", "prices.usd")
      .withColumnRenamed("usd", "label").filter(col("label").isNotNull).withColumn("label", col("label").cast("double"))
    cleanedDataDF.show()

    //handle categorical data - Create utils for this and hot indexing
    val manaCostIndexer = new StringIndexer()
      .setInputCol("mana_cost")
      .setOutputCol("mana_cost_indexed")
      .setHandleInvalid("keep") // Handle unseen labels by creating an additional category

    val loyaltyIndexer = new StringIndexer()
      .setInputCol("loyalty")
      .setOutputCol("loyalty_indexed")
      .setHandleInvalid("keep")

    val powerIndexer = new StringIndexer()
      .setInputCol("power")
      .setOutputCol("power_indexed")
      .setHandleInvalid("keep")

    val toughnessIndexer = new StringIndexer()
      .setInputCol("toughness")
      .setOutputCol("toughness_indexed")
      .setHandleInvalid("keep")

    val manaCostIndexed = manaCostIndexer.fit(cleanedDataDF).transform(cleanedDataDF)
    val loyaltyIndexed = loyaltyIndexer.fit(manaCostIndexed).transform(manaCostIndexed)
    val powerIndexed = powerIndexer.fit(loyaltyIndexed).transform(loyaltyIndexed)
    val toughnessIndexed = toughnessIndexer.fit(powerIndexed).transform(powerIndexed)

    // Final DataFrame with all indexed columns
    val indexedDataDF = toughnessIndexed
    indexedDataDF.show()

    //assemble features into a single vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("mana_cost_indexed", "game_changer", "loyalty_indexed", "power_indexed", "toughness_indexed"))
      .setOutputCol("features")

    val featureDataDF = assembler.transform(indexedDataDF).select("features", "label")
    featureDataDF.show()
    //split the data into training and test data
    val Array(trainingData, testData) = featureDataDF.randomSplit(Array(0.8, 0.2))

    //create amd train the model
    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")
    val lrModel = lr.fit(trainingData)

    //print the coefficients and intercept
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    //make and show predictions on test data
    val predictions = lrModel.transform(testData)
    predictions.select("features", "label", "prediction").show()

    //evaluate predictions
    val eval = lrModel.evaluate(testData)
    println(s"RMSE: ${eval.rootMeanSquaredError}")
    println(s"R2: ${eval.r2}")

    //stop the session
    spark.stop()
  }

}