package linearregression

import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.sql.functions.{col, explode}

object Utils {
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