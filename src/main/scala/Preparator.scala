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
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StandardScalerModel

class PreparedData(
  val rows: DataFrame,
  val dsp: DataSourceParams,
  val ssModel: StandardScalerModel
) extends Serializable

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    if (trainingData.dsp.useStandardScaler) {
      val training = trainingData.rows.toDF("label", "censor", "features")
      val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(trainingData.dsp.standardScalerWithStd).setWithMean(trainingData.dsp.standardScalerWithMean)
      val scalerModel = scaler.fit(training)
      val scaledData = scalerModel.transform(training)
      val s1 = scaledData.select("label","censor","scaledFeatures").withColumnRenamed("scaledFeatures","features")
      new PreparedData(rows = s1, dsp = trainingData.dsp, ssModel = scalerModel)
    }
    else {
      new PreparedData(rows = trainingData.rows.toDF("label", "censor", "features"), dsp = trainingData.dsp)
    }
  }
}


