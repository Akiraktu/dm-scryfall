package linearregression

import linearregression.Utils.{countValuesInColumn, getTopValues}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{array_contains, avg, col, udf}
import org.apache.spark.sql.types.StructType

object Processing {
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

}