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

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import grizzled.slf4j.Logger
import org.apache.spark.mllib.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.regression.{AFTSurvivalRegression,AFTSurvivalRegressionModel}

case class AlgorithmParams(
  val quantileProbabilities: Array[Double],
  val fitIntercept: Boolean,
  val maxIter: Int,
  val convTolerance: Double
) extends Params

class SRModel(
  val aAFTSRModel: AFTSurvivalRegressionModel,
  val ssModel: StandardScalerModel,
  val useStandardScaler: Boolean
) extends Serializable {}

class SRAlgorithm(val ap: AlgorithmParams) extends P2LAlgorithm[PreparedData, SRModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): SRModel = {
    println("Training SR model.")
    val aft = new AFTSurvivalRegression().setQuantileProbabilities(ap.quantileProbabilities).setQuantilesCol("quantiles").setFitIntercept(ap.fitIntercept).setMaxIter(ap.maxIter).setTol(ap.convTolerance)
    val model = aft.fit(data.rows)

    new SRModel(aAFTSRModel = model, ssModel=data.ssModel, useStandardScaler = data.dsp.useStandardScaler)
  }

  def predict(model: SRModel, query: Query): PredictedResult = {
    // 
    val qryRow0 = Vectors.dense(query.features)
    val qryRow = if (model.useStandardScaler) {
      model.ssModel.transform(qryRow0)
    } else {
      qryRow0
    }
    val score = model.predict(qryRow)
    val quantilesVec = model.predictQuantiles(qryRow)

    PredictedResult(coefficients = model.coefficients(),
                    intercept = model.intercept(),
                    scale = model.scale(),
                    prediction = score,
                    quantiles = quantilesVec)
  }
}
