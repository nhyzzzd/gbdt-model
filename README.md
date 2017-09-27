# gbdt-model
run a gbdt-model on spark based on the data of Avazu on kaggle

package org.apache.spark.examples.ml

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel,GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString,StringIndexer,VectorIndexer}

object gbdt {
  /* 以下程序将会输出
   */

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val conf = new SparkConf().setAppName("csvDataFrame").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import com.databricks.spark.csv._
    val data = sqlContext.csvFile(filePath = "C:\\Users\\Administrator.SZ-20170728AYLG\\IdeaProjects\\test\\data\\test.txt", useHeader = true)
    data.printSchema

    val labelIndexer = new StringIndexer()
      .setInputCol("click")
      .setOutputCol("indexedLabel")
      .fit(data)
    data.head(10).foreach(println)

    sqlContext.udf.register("hash", (str:String) => math.abs(str.hashCode))
    val data1 = data.selectExpr("indexedLabel",
      "cast(click as Int) click",
      "cast (hour as Long) hour",
      "cast (C1 as Long) C1",
      "cast (banner_pos as Int) banner_pos",
      "hash (site_id) %100 site_id",
      "hash (site_domain) %100 site_domain",
      "hash (site_category) %100 site_category",
      "hash (app_id) %100 app_id",
      "hash (app_domain) %100 app_domain",
      "hash (app_category) %100 app_category",
      "hash (device_id) %100 device_id",
      "hash (device_ip) %100 device_ip",
      "hash (device_model) %100 device_model",
      "cast (device_type as Long) device_type",
      "cast (device_conn_type as Long) device_conn_type",
      "cast (C14 as Long) C14",
      "cast (C15 as Long) C15",
      "cast (C16 as Long) C16",
      "cast (C17 as Long) C17",
      "cast (C18 as Long) C18",
      "cast (C19 as Long) C19",
      "cast (C20 as Long) C20",
      "cast (C21 as Long) C21")
    data1.show(10,false)
    data1.printSchema

    import org.apache.spark.ml.feature.VectorAssembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour",
        "C1",
        "banner_pos",
        "site_id",
        "site_domain",
        "site_category",
        "app_id",
        "app_domain",
        "app_category",
        "device_id",
        "device_ip",
        "device_model",
        "device_type",
        "device_conn_type",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21"))
      .setOutputCol("features")
    val output = assembler.transform(data1)
    output.show(10,false)
    output.printSchema()
    println(output.select("features").first())

    //train and test data
    val Array(trainingData,testData) = output.randomSplit(Array(0.7,0.3))
    //train model
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxIter(10)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer,gbt,labelConverter))

    //train model.Run the indexers.
    val model = pipeline.fit(trainingData)

    //make predictions
    val predictions = model.transform(testData)

    predictions.select("predictedLabel","click","features").show(5)

    //Select (prediction,ture label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error  =" + (1.0-accuracy))

    //get tree model
    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println("Learned classification tree model:\n" + gbtModel.toDebugString)

    sc.stop()
  }
}

