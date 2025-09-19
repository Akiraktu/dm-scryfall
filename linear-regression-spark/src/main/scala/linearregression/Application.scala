package linearregression

import jdk.jfr.Threshold
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions.{array_contains, avg, col, explode, lit, size, udf, when}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.StructType

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
      .filter(col("prices.eur").isNotNull && !col("digital") && col("type_line").contains("Creature"))
      .withColumn("label", col("prices.eur").cast("double"))
    filteredDf.select("name", "cmc", "power", "toughness", "colors", "label", "edhrec_rank", "keywords", "type_line", "legalities", "rarity").show(5, truncate = false)


    println("-> Starte CMC-Verarbeitung...")
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
      tempDf.withColumn(s"type_$typeLine", array_contains(col("type_line"), typeLine))
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

}