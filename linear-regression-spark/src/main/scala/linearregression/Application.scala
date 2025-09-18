package linearregression

import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions.{array_contains, avg, col, lit, size, udf, when}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

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
      .filter(col("prices.eur").isNotNull && !col("digital"))
      .withColumn("label", col("prices.eur").cast("double"))
    filteredDf.select("name", "cmc", "power", "toughness", "colors", "label").show(5, truncate = false)


    println("-> Starte CMC-Verarbeitung...")
    val cmcProcessedDf = processCmc(filteredDf)
    cmcProcessedDf.select("name", "cmc", "power", "label").show(5)

    val ptProcessedDf = processPowerToughness(cmcProcessedDf)
    ptProcessedDf.select("name", "cmc", "power", "toughness", "colors", "label").show(5, truncate = false)

    val colorProcessedDf = processColors(ptProcessedDf)
    colorProcessedDf.select(
      "name", "cmc", "power", "toughness", "color_W", "color_U", "color_G", "color_R", "color_B", "is_colorless",
      "identity_W", "identity_U", "identity_G", "identity_R", "identity_B", "label").show(5, truncate = false)

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

}