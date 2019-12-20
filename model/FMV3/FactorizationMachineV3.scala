package model.FMV3

import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDV}


/**
  * Factorization Machine model.
  */
class FMModelV3(val task: Int,
              val factorMatrix: Matrix,  // 二阶交叉特征的embedding：v1-vn
              val weightVector: Option[Vector],  // 一阶特征的w1-wn
              val intercept: Double,  // 零阶特征的w0
              val min: Double,
              val max: Double) extends Serializable {

  // factorMatrix(特征的维度：12， 特征的数量：一千万)
  val numFeatures = factorMatrix.numCols
  val numFactors = factorMatrix.numRows

  require(numFeatures > 0 && numFactors > 0)
  require(task == 0 || task == 1)

  def getV: Matrix = factorMatrix
  def getW: Option[Vector] = weightVector
  def getW0: Double = intercept

  def predict(testData: Vector): Double = {
    require(testData.size == numFeatures)

    var pred = intercept
    if (weightVector.isDefined) {
      testData.foreachActive {
        case (i, v) =>
          pred += weightVector.get(i) * v
      }
    }

    for (f <- 0 until numFactors) {
      var sum = 0.0
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          val d = factorMatrix(f, i) * v
          sum += d
          sumSqr += d * d
      }
      pred += (sum * sum - sumSqr) * 0.5
    }

    task match {
      case 0 =>
        Math.min(Math.max(pred, min), max)
      case 1 =>
        1.0 / (1.0 + Math.exp(-pred))
    }
  }

  def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions {
      _.map {
        vec =>
          predict(vec)
      }
    }
  }

}



class FMGradientV3(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
                 val numFeatures: Int, val min: Double, val max: Double,
                 val r0: Double, val r1: Double, val r2: Double) extends Serializable {

  def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {
    // 注意，对于分类问题，这里的pred是预测的概率值，不是wx

    var pred = if (k0) weights(weights.size - 1) else 0.0

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          pred += weights(pos + i) * v
      }
    }

    val sum = Array.fill(k2)(0.0)  // sum = v1x1 + v2x2 + v3x3 + ... vnxn
    for (f <- 0 until k2) {
      var sumSqr = 0.0
      data.foreachActive {
        case (i, v) =>
          val d = weights(i * k2 + f) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) * 0.5
    }

    if (task == 0) {
      pred = Math.min(Math.max(pred, min), max)
    } else {
      pred = 0.5 * (1 + Math.tanh(0.5 * pred))  // 注：这里是预测值 0～1
    }

    (pred, sum)
  }


  // SGD
  def computeFM(data: Vector, label: Double, weights: Vector, stepSize: Double, iter: Int): BDV[Double] = {

    require(data.size == numFeatures)

    val (pred, sum) = predict(data, weights)

    // partial(Loss) / partial(wx)
    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        // Loss = -1 * cross-entropy = -ylog(p) - (1-y)log(1-p)
        // partial(p) / partial(wx) = p(1-p)
        if (label >= Double.MinPositiveValue)  // -log(p)
          pred - 1.0  // partial(Loss) / partial(wx) = p-1
        else  // -log(1-p)
          pred  // partial(Loss) / partial(wx) = p
    }

    val thisIterStepSize = stepSize// / math.sqrt(iter)  // 暂时不随迭代轮数减小学习率
    val len = weights.size
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    var grad = 0.0

    if (k0) {
      grad = mult + r0 * weights(len - 1)
      weightsArray(len - 1) = weights(len - 1) - thisIterStepSize * grad / Math.sqrt(grad * grad + 1.0)
      if (weightsArray(len - 1) > 20.0) weightsArray(len - 1) = 20.0
      else if (weightsArray(len - 1) < -20.0) weightsArray(len - 1) = -20.0
    }

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          require(v == 1.0)
          grad = v * mult + r1 * weightsArray(pos + i)
          weightsArray(pos + i) -= thisIterStepSize * grad / Math.sqrt(grad * grad + 1.0)
          if (weightsArray(pos + i) > 20.0) weightsArray(pos + i) = 20.0
          else if (weightsArray(pos + i) < -20.0) weightsArray(pos + i) = -20.0
      }
    }

    data.foreachActive {  // sum = v1x1 + v2x2 + v3x3 + ... vnxn
      case (i, v) =>
        val pos = i * k2
        for (f <- 0 until k2) {
          require(v == 1.0)
          grad = (sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + f)
          weightsArray(pos + f) -= thisIterStepSize * grad / Math.sqrt(grad * grad + 1.0)
          if (weightsArray(pos + f) > 20.0) weightsArray(pos + f) = 20.0
          else if (weightsArray(pos + f) < -20.0) weightsArray(pos + f) = -20.0
        }
    }

    BDV(weightsArray)
  }

  // Adam
  def computeFM(data: Vector, label: Double, weights: Vector, adamG1: Vector, adamG2: Vector,
                stepSize: Double, iter: Int): (BDV[Double], BDV[Double], BDV[Double]) = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)

    // partial L / partial pred
    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
    }

    var g = 0.0
    val thisIterStepSize = stepSize// / math.sqrt(iter)  // 暂时不随迭代轮数减小学习率
    val rho1 = 0.9
    val rho2 = 0.999
    val eps = 1e-8
    val len = weights.size
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val g1Array: Array[Double] = adamG1.asInstanceOf[DenseVector].values
    val g2Array: Array[Double] = adamG2.asInstanceOf[DenseVector].values

    if (k0) {
      g = mult + r0 * weights(len - 1)
      g1Array(len - 1) = rho1 * g1Array(len - 1) + (1.0 - rho1) * g
      g2Array(len - 1) = rho2 * g2Array(len - 1) + (1.0 - rho2) * g * g
      weightsArray(len - 1) = weights(len - 1) - thisIterStepSize * g1Array(len - 1) / (1.0 - rho1) / (Math.sqrt(g2Array(len - 1) / (1.0 - rho2)) + eps)
    }

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          g = v * mult + r1 * weightsArray(pos + i)
          g1Array(pos + i) = rho1 * g1Array(pos + i) + (1.0 - rho1) * g
          g2Array(pos + i) = rho2 * g2Array(pos + i) + (1.0 - rho2) * g * g
          weightsArray(pos + i) -= thisIterStepSize * g1Array(pos + i) / (1.0 - rho1) / (Math.sqrt(g2Array(pos + i) / (1.0 - rho2)) + eps)
      }
    }

    data.foreachActive {  // sum = v1x1 + v2x2 + v3x3 + ... vnxn
      case (i, v) =>
        val pos = i * k2
        for (f <- 0 until k2) {
          g = (sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + f)
          g1Array(pos + f) = rho1 * g1Array(pos + f) + (1.0 - rho1) * g
          g2Array(pos + f) = rho2 * g2Array(pos + f) + (1.0 - rho2) * g * g
          //weightsArray(pos + i) -= thisIterStepSize * ((sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + i))
          weightsArray(pos + f) -= thisIterStepSize * g1Array(pos + f) / (1.0 - rho1) / (Math.sqrt(g2Array(pos + f) / (1.0 - rho2)) + eps)
        }
    }

    (BDV(weightsArray), BDV(g1Array), BDV(g2Array))
  }

  // RMSProp
  def computeFM(data: Vector, label: Double, weights: Vector, rmspropG: Vector,
                stepSize: Double, iter: Int): (BDV[Double], BDV[Double]) = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)

    // partial L / partial pred
    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        val alpha = 1.0//val alpha = if (label > Double.MinPositiveValue) 5.0 else 1.0
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred))) * alpha
    }

    var g = 0.0
    val thisIterStepSize = stepSize// / math.sqrt(iter)  // 暂时不随迭代轮数减小学习率
    val rho = 0.99
    val eps = 1e-8
    val len = weights.size
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val gArray: Array[Double] = rmspropG.asInstanceOf[DenseVector].values

    if (k0) {
      g = mult + r0 * weights(len - 1)
      gArray(len - 1) = rho * gArray(len - 1) + (1.0 - rho) * g * g
      weightsArray(len - 1) = weights(len - 1) - thisIterStepSize * g / Math.sqrt(gArray(len - 1) + eps)
    }

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          g = v * mult + r1 * weightsArray(pos + i)
          gArray(pos + i) = rho * gArray(pos + i) + (1.0 - rho) * g * g
          weightsArray(pos + i) -= thisIterStepSize * g / Math.sqrt(gArray(pos + i) + eps)
      }
    }

    data.foreachActive {  // sum = v1x1 + v2x2 + v3x3 + ... vnxn
      case (i, v) =>
        val pos = i * k2
        for (f <- 0 until k2) {
          g = (sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + f)
          gArray(pos + f) = rho * gArray(pos + f) + (1.0 - rho) * g * g
          //weightsArray(pos + i) -= thisIterStepSize * ((sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + i))
          weightsArray(pos + f) -= thisIterStepSize * g / Math.sqrt(gArray(pos + f) + eps)
        }
    }

    (BDV(weightsArray), BDV(gArray))
  }
}
