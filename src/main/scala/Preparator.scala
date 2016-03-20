package org.template.sr

/*
 * Copyright KOLIBERO under one or more contributor license agreements.  
 * KOLIBERO licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StandardScaler
//import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StandardScalerModel
//import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors

class PreparedData(
  val rows: DataFrame,
  val dsp: DataSourceParams,
  val ssModel: org.apache.spark.mllib.feature.StandardScalerModel
) extends Serializable

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    if (trainingData.dsp.useStandardScaler) {
      val training = trainingData.rows.map(x=>(x._1,x._2,Vectors.dense(x._3))).toDF("label", "censor", "features")
      val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(trainingData.dsp.standardScalerWithStd).setWithMean(trainingData.dsp.standardScalerWithMean)
      val scalerModel = scaler.fit(training)
      val scaledData = scalerModel.transform(training)
      val s1 = scaledData.select("label","censor","scaledFeatures").withColumnRenamed("scaledFeatures","features")

      //Prepare old StandardScaler
      val oldScaler = new org.apache.spark.mllib.feature.StandardScaler(withMean = trainingData.dsp.standardScalerWithMean, withStd = trainingData.dsp.standardScalerWithStd)
      val oldSSModel = oldScaler.fit(trainingData.rows.map(x=>(Vectors.dense(x._3))))
            
      new PreparedData(rows = s1, dsp = trainingData.dsp, ssModel = oldSSModel)
    }
    else {
      new PreparedData(rows = trainingData.rows.map(x=>(x._1,x._2,Vectors.dense(x._3))).toDF("label", "censor", "features"), dsp = trainingData.dsp, ssModel = null)
    }
  }
}


