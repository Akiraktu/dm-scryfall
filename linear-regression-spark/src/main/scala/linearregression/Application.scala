package linearregression

import jdk.jfr.Threshold
import org.apache.hadoop.shaded.com.google.common.base.Functions
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions.{array_contains, avg, col, explode, lit, rand, round, abs,  size, sum, udf, when}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.types.StructType

object Application {
  def main(args: Array[String]): Unit = {

    //initialize spark session using all cores
    val spark = SparkSession.builder().appName("Price Prediction")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    //load data
    val dataPath = "src/main/resources/scryfall-oracle-cards.json"
    val scryfallDataDF = spark.read.json(dataPath)

    val filteredDf = scryfallDataDF
      .filter(col("prices.eur").isNotNull && !col("digital") && col("type_line").contains("Creature") && col("card_faces").isNull)
      .withColumn("price", col("prices.eur").cast("double")).select("name", "cmc", "power", "toughness", "colors", "color_identity", "price", "edhrec_rank", "keywords", "type_line", "legalities", "rarity")
    filteredDf.show(5)


    println("-> Starting feature processing...")
    println("-> Step: Process CMC")
    val cmcProcessedDf = processCmc(filteredDf)
    cmcProcessedDf.show(5)

    println("-> Step: Process Power and Toughness")
    val ptProcessedDf = processPowerToughness(cmcProcessedDf)
    ptProcessedDf.show(5)

    println("-> Step: Process Colors and Color Identity")
    val colorProcessedDf = processColors(ptProcessedDf)
    colorProcessedDf.show(5)

    println("-> Step: Process EDHREC Rank")
    val edhrecProcessedDf = processEdhrecRank(colorProcessedDf)
    edhrecProcessedDf.show(5)

    println("-> Step: Process Keywords")
    val keywordProcessedDf = processKeywords(edhrecProcessedDf, 50)
    keywordProcessedDf.show(5)

    println("-> Step: Process Type Line")
    val typeLineProcessedDf = processTypeLine(keywordProcessedDf, 120)
    typeLineProcessedDf.show(5)

    println("-> Step: Process Legalities")
    val legalitiesProcessedDf = processLegalities(typeLineProcessedDf)
    legalitiesProcessedDf.show(5)

    println("-> Step: Process Rarity")
    val rarityProcessedDf = processRarity(legalitiesProcessedDf)

    //get all the color, type, keyword and legality columns
    val colorColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "color_").filter(col => ( col != "color_identity") && (col != "color_indicator"))
    val identityColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "identity_")
    val keywordColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "keyword_")
    val typeColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "types_")
    val legalityColumns = getAllRowsWithPrefix(legalitiesProcessedDf, "legality_")

    //build the array of feature columns
    val featureColumns = Array("cmc", "power", "toughness", "edhrec_rank", "rarityEncoded", "is_colorless") ++ colorColumns ++ identityColumns ++ keywordColumns ++ typeColumns ++ legalityColumns

    rarityProcessedDf.select((Array("name", "price") ++ featureColumns).map(col): _*).show(5)

    //assemble features into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val assembledDf = assembler.transform(rarityProcessedDf).select("name", "features", "price")

    //split data into training and test sets
    val Array(trainingData, testData) = assembledDf.randomSplit(Array(0.8, 0.2), seed = 84)

    //create a random forest regressor
    val forestRegressor = new RandomForestRegressor()
      .setLabelCol("price")
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
      .setLabelCol("price")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val r2Evaluator = new RegressionEvaluator()
      .setLabelCol("price")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val rmse = rmseEvaluator.evaluate(predictions)
    val r2 = r2Evaluator.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE): $rmse")
    println(s"R²: $r2")

    //show the actual and predicted prices for the test data
    predictions.withColumn("prediction", round(col("prediction"), 2)).alias("p").join(testData.alias("t"), "name")
      .orderBy(rand()).select("name", "p.price", "p.prediction")
      .withColumn("difference", round(col("p.price") - col("p.prediction"), 2))
      .show(20)

    //stop the session
    spark.stop()
  }

  def processCmc(df: DataFrame): DataFrame = {
    // ensure type is correct
    val dfWithTypedCmc = df.withColumn("cmc", col("cmc").cast("double"))
    // calculate avg ignoring null vals
    val cmcAvg = dfWithTypedCmc.select(avg("cmc")).first().getDouble(0)
    // fill null values with avg
    dfWithTypedCmc.na.fill(cmcAvg, Seq("cmc"))
  }

  def processPowerToughness(df: DataFrame): DataFrame = {
    // udf to change strings that only have chars into ints
    // gives back 0 for everything else (X", "*", "1+*" etc)
    val toIntUDF = udf((s: String) => {
      if (s != null && s.matches("\\d+")) Some(s.toInt) else None: Option[Int]
    })

    // create new numeric columns
    val dfWithNumeric = df
      .withColumn("power", toIntUDF(col("power")))
      .withColumn("toughness", toIntUDF(col("toughness")))

    // calculate averages
    val powerAvg = dfWithNumeric.select(avg("power")).first().getDouble(0)
    val toughnessAvg = dfWithNumeric.select(avg("toughness")).first().getDouble(0)

    // fill null vals with avg
    dfWithNumeric.na.fill(Map(
      "power" -> powerAvg,
      "toughness" -> toughnessAvg
    ))
    }

  def processColors(df: DataFrame): DataFrame = {
    val manaColors = Seq("W", "U", "B", "R", "G")

    // creates a column for each color and color identity
    val dfWithColorFlags = manaColors.foldLeft(df) { (tempDf, color) =>
      tempDf
        .withColumn(s"identity_$color", array_contains(col("color_identity"), color))
        .withColumn(s"color_$color", array_contains(col("colors"), color))
    }

    // creates a condition for "is_colorless"
    // a card is colorless when all "color_X" columns are false
    val colorlessCondition = manaColors
      .map(color => col(s"color_$color") === false)
      .reduce((cond1, cond2) => cond1 && cond2)

    dfWithColorFlags.withColumn("is_colorless", colorlessCondition)
  }

  def processEdhrecRank(df: DataFrame): DataFrame = {
    // fill null values with 0
    df.na.fill(0, Seq("edhrec_rank"))
  }

  def processKeywords(df: DataFrame, threshold: Int): DataFrame = {
    // count distinct keywords
    val distinctKeywords = countValuesInColumn(df, "keywords")

    // create a column for each of the keywords in the threshold
    val topKeywords = getTopValues(distinctKeywords, "keywords", threshold)

    topKeywords.foldLeft(df) { (tempDf, keyword) =>
      tempDf.withColumn(s"keyword_$keyword", array_contains(col("keywords"), keyword))
    }
  }

  def processTypeLine(df: DataFrame, threshold: Int): DataFrame = {
    //udf to turn the type line into an array of strings
    val toArrayUdf = udf((s: String) => {if (s != null) s.split("[^A-Za-z]+").map(_.trim) else Array.empty[String]})

    //create a new column with the type line as array
    val dfWithTypeArray = df.withColumn("type_line", toArrayUdf(col("type_line")))

    //get disctinct types and their counts
    val distinctTypes = countValuesInColumn(dfWithTypeArray, "type_line")

    //create a column for each of the types in the threshold
    val topTypes = getTopValues(distinctTypes, "type_line", threshold)

    topTypes.foldLeft(dfWithTypeArray) { (tempDf, typeLine) =>
      tempDf.withColumn(s"types_$typeLine", array_contains(col("type_line"), typeLine))
    }

  }

  def processLegalities(df: DataFrame): DataFrame = {
    //udf for checking legality
    val isLegalUdf = udf((status: String) => if (status == "legal") true else false)

    //get formats
    val structType = df.schema("legalities").dataType.asInstanceOf[StructType]
    val legalities = structType.fieldNames

    //create a column for each legality
    legalities.foldLeft(df) { (tempDf, legality) =>
      tempDf.withColumn(s"legality_$legality", isLegalUdf(col(s"legalities.$legality")))
    }
  }

  def processRarity(df: DataFrame): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol("rarity")
      .setOutputCol("rarityIndex")
      .setHandleInvalid("keep")

    val indexedDF = indexer.fit(df).transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol("rarityIndex")
      .setOutputCol("rarityEncoded")
      .setDropLast(false)

    encoder.fit(indexedDF).transform(indexedDF)
  }

  def countValuesInColumn(df: DataFrame, colName: String): DataFrame = {
    //count the values in the column and order them descending
    df.withColumn(colName, explode(col(colName))).groupBy(col(colName)).count().orderBy(functions.desc("count"))
  }

  def getTopValues(df: DataFrame, colName: String, threshold: Int): Seq[String] = {
    if (threshold == -1) //-1 means all values
      df.select(colName).collect().map(row => row.getString(0))
    else
      df.limit(threshold).select(colName).collect().map(row => row.getString(0))
  }

  def getAllRowsWithPrefix(df: DataFrame, prefix: String): Array[String] = {
    df.columns.filter(_.startsWith(prefix))
  }

}