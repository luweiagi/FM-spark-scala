package model.FMV3

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{norm, DenseVector => BDV}
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent}
import org.apache.spark.storage.StorageLevel
//import org.apache.spark.mllib.regression.FMGradient
import com.immomo.nearby.people.model.FMV3.FMGradientV3
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.{Gradient, Updater, Optimizer}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import java.text.SimpleDateFormat


class GradientDescentFMV3(private val gradient: FMGradientV3, private val opt: String,
                          private val validData: RDD[(Double, Vector)], private val stepSize: Double,
                          private var numIterations: Int, private var numPartition: Int) extends Serializable {

  //private var regParam: Double = 0.0

  def optimize(trainData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescentFMV3.parallelSGD(
      trainData,
      validData,
      gradient,
      stepSize,
      numIterations,
      opt,
      initialWeights,
      numPartition)

    weights
  }
}


object GradientDescentFMV3 {

  def breezevec2vec(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }


  def evaluateLogLossAndAUC(data: RDD[(Double, Vector)], weights: Vector, gradient: FMGradientV3): (Double, Double, Double, Double) = {
    /*  Evaluate a Factorization Machine model on a data set.*/
    val bcWeights = data.context.broadcast(weights)

    println("==========evaluateLogLossAndAUC=============")
    println("weight size = " + weights.size)
    print("weights = ")
    weights.toArray.slice(0,20).foreach(x => print(x + " | ")); println("")
    weights.toArray.slice(weights.size - 21, weights.size - 1).foreach(x => print(x + " | ")); println("")

    val max = weights.toArray.max//foreach(x => {if(x > max) max = x; if(x < min) min = x})
    val min = weights.toArray.min//foreach(x => {if(x > max) max = x; if(x < min) min = x})
    println("max = " + max + "; min = " + min)
    // 如果出现NAN值，则退出
    weights.toArray.foreach(x => if (x.isNaN) {println("weight exits NaN! exit!"); System.exit(-1)})

    val dataVector: RDD[Vector] = data.map(label_vector => label_vector._2)
    val labelSample: RDD[Double] = data.map(label_vector => label_vector._1).persist(StorageLevel.MEMORY_AND_DISK)

    val labelPredict: RDD[Double] = dataVector.map(
      x =>
        gradient.asInstanceOf[FMGradientV3].predict(x, bcWeights.value) // y预测
    ).map(
      x =>
        x._1
    ).persist(StorageLevel.MEMORY_AND_DISK)

    println("labelPredict: ")
    labelPredict.take(10).foreach(println)


    // logLoss
    val (logLossSum, logLossCount) = labelPredict.zip(
      labelSample
    ).map(
      x =>
        if (x._2 > Double.MinPositiveValue) {  // y = 1
          -1.0 * Math.log(x._1)
        } else {  // y = 0
          -1.0 * Math.log(1.0 - x._1)
        }
    ).treeAggregate((0d, 0))(
      (acc, value) => (acc._1 + value, acc._2 + 1),
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
    )

    val logLoss = logLossSum / logLossCount.toDouble


    // AUC
    val scoreAndLabels: RDD[(Double, Double)] = labelPredict.zip(labelSample)

    scoreAndLabels.take(10).foreach(println)

    val bmetrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = bmetrics.areaUnderROC


    // (正样本的预测得分加和, 正样本的总数, 负样本的预测得分加和, 负样本的总数)
    val scoresPredict: (Double, Int, Double, Int) = labelSample.zip(
      labelPredict
    ).treeAggregate((0d, 0, 0d, 0))(
      (acc, value) => (
        if(value._1 > Double.MinPositiveValue) acc._1 + value._2 else acc._1,
        if(value._1 > Double.MinPositiveValue) acc._2 + 1 else acc._2,
        if(value._1 < Double.MinPositiveValue) acc._3 + value._2 else acc._3,
        if(value._1 < Double.MinPositiveValue) acc._4 + 1 else acc._4
      ),
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2, acc1._3 + acc2._3, acc1._4 + acc2._4)
    )

    val posPredict = scoresPredict._1 / scoresPredict._2.toDouble
    val negPredict = scoresPredict._3 / scoresPredict._4.toDouble

    (logLoss, auc, posPredict, negPredict)
  }


  // 用于训练数据的随机shuffle
  /*import org.apache.spark.Partitioner
  class randomPartitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      scala.util.Random.nextInt(numPartitions)
    }
  }*/

  def parallelSGD(
                  trainData: RDD[(Double, Vector)],
                  validData: RDD[(Double, Vector)],
                  gradient: FMGradientV3,
                  stepSize: Double,
                  numIterations: Int,
                  opt: String,
                  initialWeights: Vector,
                  numPartition: Int): (Vector, Array[String]) = {

    // Record previous weight and current one to calculate solution vector difference
    val stochasticLossHistory = new ArrayBuffer[String]()

    if (trainData.count() == 0) {
      println("No data found! trainData.count() = 0")
      System.exit(-1)
    }

    println("numPartition = " + numPartition)

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    println("weights.size = " + n)

    var adamG1 = Vectors.dense(Array.fill[Double](0)(0.0))
    var adamG2 = Vectors.dense(Array.fill[Double](0)(0.0))
    var rmspropG = Vectors.dense(Array.fill[Double](0)(0.0))

    println("opt: " + opt)
    opt match {
      case "adam" =>
        adamG1 = Vectors.dense(Array.fill[Double](n)(0.0))
        adamG2 = Vectors.dense(Array.fill[Double](n)(0.0))
      case "rmsprop" =>
        rmspropG = Vectors.dense(Array.fill[Double](n)(0.0))
      case _ => ;
    }

    val slices = trainData.getNumPartitions
    println("slices = " + slices)

    val df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")


    println("epoch    trainLogLoss    validLogLoss    trainAUC    validAUC    trainPosPredict    trainNegPredict    validPosPredict    validNegPredict    timePeriod(min)    date")
    val (trainLogLoss, trainAUC, trainPosPredict, trainNegPredict) = evaluateLogLossAndAUC(trainData, weights, gradient)
    val (validLogLoss, validAUC, validPosPredict, validNegPredict) = evaluateLogLossAndAUC(validData, weights, gradient)
    var evaluateInfo = "%02d       %.5f         %.5f         %.5f     %.5f     %.5f            %.5f            %.5f            %.5f            None               %s".format(
      0, trainLogLoss, validLogLoss, trainAUC, validAUC, trainPosPredict, trainNegPredict, validPosPredict, validNegPredict, df.format(System.currentTimeMillis()))
    println(evaluateInfo)
    stochasticLossHistory.+=(evaluateInfo)


    var i = 1
    while (i <= numIterations) {

      val timeBegin = System.currentTimeMillis()

      val bcWeights = trainData.context.broadcast(weights)

      opt match {
        case "sgd" =>
          val wSum = trainData.treeAggregate(BDV(bcWeights.value.toArray))(
            seqOp = (c, v) => {
              gradient.asInstanceOf[FMGradientV3].computeFM(v._2, v._1, breezevec2vec(c), stepSize, i)
            },
            combOp = (c1, c2) => {
              c1 + c2
            },
            7
          )
          weights = Vectors.dense(wSum.toArray.map(_ / numPartition))

        case "adam" =>
          val bcAdamG1 = trainData.context.broadcast(adamG1)
          val bcAdamG2 = trainData.context.broadcast(adamG2)
          val (wSum, g1Sum, g2Sum) = trainData.treeAggregate((BDV(bcWeights.value.toArray), BDV(bcAdamG1.value.toArray), BDV(bcAdamG2.value.toArray)))(
            seqOp = (c, v) => {
              gradient.asInstanceOf[FMGradientV3].computeFM(v._2, v._1, breezevec2vec(c._1), breezevec2vec(c._2), breezevec2vec(c._3), stepSize, i)
            },
            combOp = (c1, c2) => {
              (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
            },
            7
          )
          weights = Vectors.dense(wSum.toArray.map(_ / numPartition))
          adamG1 = Vectors.dense(g1Sum.toArray.map(_ / numPartition))
          adamG2 = Vectors.dense(g2Sum.toArray.map(_ / numPartition))

        case "rmsprop" =>
          val bcRmspropG = trainData.context.broadcast(rmspropG)
          val (wSum, gSum) = trainData.treeAggregate((BDV(bcWeights.value.toArray), BDV(bcRmspropG.value.toArray)))(
            seqOp = (c, v) => {
              gradient.asInstanceOf[FMGradientV3].computeFM(v._2, v._1, breezevec2vec(c._1), breezevec2vec(c._2), stepSize, i)
            },
            combOp = (c1, c2) => {
              (c1._1 + c2._1, c1._2 + c2._2)
            },
            7
          )
          weights = Vectors.dense(wSum.toArray.map(_ / numPartition))
          rmspropG = Vectors.dense(gSum.toArray.map(_ / numPartition))

        case _ =>
          println("error: no opt type match! exit!")
          System.exit(-1)
      }

      //trainData.partitionBy(new randomPartitioner(numPartition))

      val (trainLogLoss, trainAUC, trainPosPredict, trainNegPredict) = evaluateLogLossAndAUC(trainData, weights, gradient)
      val (validLogLoss, validAUC, validPosPredict, validNegPredict) = evaluateLogLossAndAUC(validData, weights, gradient)

      val timeEnd = System.currentTimeMillis()

      evaluateInfo = "%02d       %.5f         %.5f         %.5f     %.5f     %.5f            %.5f            %.5f            %.5f            %.1f               %s".format(
        i, trainLogLoss, validLogLoss, trainAUC, validAUC, trainPosPredict, trainNegPredict, validPosPredict, validNegPredict, (timeEnd - timeBegin).toDouble/1000d/60d, df.format(System.currentTimeMillis()))
      println(evaluateInfo)
      stochasticLossHistory.+=(evaluateInfo)

      i += 1
    }

    (weights, stochasticLossHistory.toArray)
  }
}
