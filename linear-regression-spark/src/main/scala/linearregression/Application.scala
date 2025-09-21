package linearregression

import linearregression.Processing._
import linearregression.Utils.{countValuesInColumn, getAllRowsWithPrefix}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.functions.{col, sum, udf}
import org.apache.spark.sql.{SparkSession, functions}

object Application {
  def main(args: Array[String]): Unit = {

    //initialize spark session using all cores
    val spark = SparkSession.builder().appName("Price Prediction")
      .master("local[*]")
      .getOrCreate()

    //load data
    val dataPath = "src/main/resources/scryfall-oracle-cards.json"
    val scryfallDataDF = spark.read.json(dataPath)

    val filteredDf = scryfallDataDF
      .filter(col("prices.eur").isNotNull && !col("digital") && col("type_line").contains("Creature") && col("card_faces").isNull)
      .withColumn("label", col("prices.eur").cast("double"))
    filteredDf.select("name", "cmc", "power", "toughness", "colors", "label", "edhrec_rank", "keywords", "type_line", "legalities", "rarity").show(5, truncate = false)


    println("-> Starting feature processing...")
    val cmcProcessedDf = processCmc(filteredDf)
    cmcProcessedDf.select("name", "cmc", "power", "label").show(5)

    val ptProcessedDf = processPowerToughness(cmcProcessedDf)
    ptProcessedDf.select("name", "cmc", "power", "toughness", "colors", "label").show(5, truncate = false)

    val colorProcessedDf = processColors(ptProcessedDf)
    colorProcessedDf.select(
      "name", "cmc", "power", "toughness", "color_W", "color_U", "color_G", "color_R", "color_B", "is_colorless",
      "identity_W", "identity_U", "identity_G", "identity_R", "identity_B", "label").show(5, truncate = false)

    val edhrecProcessedDf = processEdhrecRank(colorProcessedDf)
    val nullValsBefore = colorProcessedDf.filter(col("edhrec_rank").isNull).count()
    val nullValsAfter = edhrecProcessedDf.filter(col("edhrec_rank").isNull).count()
    println(s"Null values in edhrec_rank \nbefore: $nullValsBefore \nafter: $nullValsAfter")
    edhrecProcessedDf.select("name", "edhrec_rank", "rarity", "keywords").show(5, truncate = false)


    val distinctKeywords = countValuesInColumn(edhrecProcessedDf, "keywords")
    println(s"Distinct keywords and their counts: ${distinctKeywords.count()}")
    distinctKeywords.orderBy(functions.rand()).show()
    val keywordProcessedDf = processKeywords(edhrecProcessedDf, 50)
    keywordProcessedDf.show(5, truncate = false)

    //udf to turn the type line into an array of strings
    val toArrayUdf = udf((s: String) => {if (s != null) s.split("[^A-Za-z]+").map(_.trim) else Array.empty[String]})
    //create a new column with the type line as array
    val dfWithTypeArray = keywordProcessedDf.withColumn("type_line", toArrayUdf(col("type_line")))
    //get disctinct types and their counts
    val distinctTypes = countValuesInColumn(dfWithTypeArray, "type_line")
    println(s"Distinct types and their counts: ${distinctTypes.count()}")
    distinctTypes.sort(functions.rand()).show()

    val typeLineProcessedDf = processTypeLine(keywordProcessedDf, 120)
    typeLineProcessedDf.show(5, truncate = false)

    val legalitiesProcessedDf = processLegalities(typeLineProcessedDf)
    legalitiesProcessedDf.show(5, truncate = false)

    val columns = legalitiesProcessedDf.columns
    val nullCounts = legalitiesProcessedDf.select(columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*)
    nullCounts.show()

    val rarityProcessedDf = processRarity(legalitiesProcessedDf)

    //get all the color, type, keyword and legality columns
    val colorColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "color_").filter(col => ( col != "color_identity") && (col != "color_indicator"))
    val identityColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "identity_")
    val keywordColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "keyword_")
    val typeColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "types_")
    val legalityColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "legality_")

    //build the array of feature columns
    val featureColumns = Array("cmc", "power", "toughness", "edhrec_rank", "rarityEncoded", "is_colorless") ++ colorColumns ++ identityColumns ++ keywordColumns ++ typeColumns ++ legalityColumns

    rarityProcessedDf.select((Array("name", "label") ++ featureColumns).map(col): _*).show(5, truncate = false)

    //check types of feature columns
    val featureColumnTypes = rarityProcessedDf.select(featureColumns.map(col): _*).dtypes
    featureColumnTypes.foreach { case (name, dataType) =>
      println(s"Column: $name, Type: $dataType")
    }

    //assemble features into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val assembledDf = assembler.transform(rarityProcessedDf).select("name", "features", "label")

    //split data into training and test sets
    val Array(trainingData, testData) = assembledDf.randomSplit(Array(0.8, 0.2))

    //create a random forest regressor
    val forestRegressor = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(10)
      .setSeed(42)

    //train the model
    println("-> Training the model...")
    val model = forestRegressor.fit(trainingData)

    //evaluate the model
    println("-> Evaluating the model...")
    val predictions = model.transform(testData)

    val rmseEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val r2Evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val rmse = rmseEvaluator.evaluate(predictions)
    val r2 = r2Evaluator.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE): $rmse")
    println(s"R²: $r2")

    //stop the session
    spark.stop()
  }
}