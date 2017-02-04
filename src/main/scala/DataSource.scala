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

import org.apache.predictionio.controller.PDataSource
import org.apache.predictionio.controller.EmptyEvaluationInfo
import org.apache.predictionio.controller.EmptyActualResult
import org.apache.predictionio.controller.Params
import org.apache.predictionio.data.store.PEventStore
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(
  val appName: String, 
  val useStandardScaler: Boolean,
  val standardScalerWithStd: Boolean,
  val standardScalerWithMean: Boolean 
) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    println("Gathering data from event server.")
    val rowsRDD: RDD[(Double, Double, Array[Double])] = PEventStore.find(
      appName = dsp.appName,
      entityType = Some("row"),
      startTime = None,
      eventNames = Some(List("$set")))(sc).map { event =>
        try {
	        (event.properties.get[Double]("label"), event.properties.get[Double]("censor"), event.properties.get[Array[Double]]("features"))
        } catch {
          case e: Exception => {
            logger.error(s"Failed to convert event ${event} of. Exception: ${e}.")
            throw e
          }
        }
      }
    new TrainingData(rowsRDD, dsp)
  }
}

class TrainingData(
  val rows: RDD[(Double, Double, Array[Double])],
  val dsp: DataSourceParams
) extends Serializable
