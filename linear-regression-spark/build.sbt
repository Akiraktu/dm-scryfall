ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.6"

lazy val root = (project in file("."))
  .settings(
    name := "Scryfall linear regression" ,
    version := "1.0",
    scalaVersion := "2.12.18",
    Compile / run / mainClass := Some("linearregression.Application"),
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.6",

  )
