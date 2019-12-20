package model.FMV3

import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.optimization.{Updater}
import com.immomo.nearby.people.model.FMV2.{GradientDescentFM}
import org.apache.spark.rdd.RDD
import scala.util.Random
import org.apache.spark.mllib.regression.{LabeledPoint}

object FMWithSGDV3 {

  def train(trainData: RDD[LabeledPoint],
            validData: RDD[LabeledPoint],
            task: Int,
            numIterations: Int,
            opt: String,
            stepSize: Double,
            dim: (Boolean, Boolean, Int),
            regParam: (Double, Double, Double),
            initStd: Double,
            numPartition: Int): FMModelV3 = {

    new FMWithSGDV3(task, stepSize, numIterations, opt, dim, regParam, initStd, numPartition).run(trainData, validData)
  }
}


class FMWithSGDV3(private var task: Int,
                  private var stepSize: Double,
                  private var numIterations: Int,
                  private var opt: String,
                  private var dim: (Boolean, Boolean, Int),
                  private var regParam: (Double, Double, Double),
                  private var initStd: Double,
                  private var numPartition: Int) extends Serializable {

  def this() = this(1, 0.1, 50, "sgd", (true, true, 8), (0, 1e-3, 1e-4), 0.1, 256)

  private val k0: Boolean = dim._1
  private val k1: Boolean = dim._2
  private val k2: Int = dim._3

  private val r0: Double = regParam._1
  private val r1: Double = regParam._2
  private val r2: Double = regParam._3

  private val initMean: Double = 0

  private var numFeatures: Int = -1
  private var minLabel: Double = Double.MaxValue
  private var maxLabel: Double = Double.MinValue


  def run(trainData: RDD[LabeledPoint], validData: RDD[LabeledPoint]): FMModelV3 = {

    val numFeaturesMayLess = trainData.first().features.size
    this.numFeatures = trainData.map(
      x =>
        x.features.size
    ).treeAggregate(0)(
      seqOp = (c, v) => {
        Math.max(c ,v)
      },
      combOp = (c1, c2) => {
        Math.max(c1, c2)
      },
      7
    )
    println("numFeaturesMayLess = " + numFeaturesMayLess)
    println("numFeatures = " + this.numFeatures)
    require(this.numFeatures > 0)

    if (task == 0) {
      val (minT, maxT) = trainData.map(_.label).aggregate[(Double, Double)]((Double.MaxValue, Double.MinValue))(
        { case ((min, max), v) =>
          (Math.min(min, v), Math.max(max, v))
        }, {
          case ((min1, max1), (min2, max2)) =>
            (Math.min(min1, min2), Math.max(max1, max2))
        })

      this.minLabel = minT
      this.maxLabel = maxT
    }


    val trainLabelFeat = task match {
      case 0 =>
        trainData.map(l => (l.label, l.features))
      case 1 =>
        trainData.map(l => (if (l.label > Double.MinPositiveValue) 1.0 else 0.0, l.features))
    }

    val validLabelFeat = task match {
      case 0 =>
        validData.map(l => (l.label, l.features))
      case 1 =>
        validData.map(l => (if (l.label > Double.MinPositiveValue) 1.0 else 0.0, l.features))
    }


    val initWeights: Vector = generateInitWeights()

    // 核心的梯度下降算法
    val gradient = new FMGradientV3(task, k0, k1, k2, numFeatures, minLabel, maxLabel, r0, r1, r2)

    // 并行进行梯度下降，调用了gradient中的核心梯度下降算法
    val optimizer = new GradientDescentFMV3(gradient, opt, validLabelFeat, stepSize, numIterations, numPartition)

    val weights = optimizer.optimize(trainLabelFeat, initWeights)

    createModel(weights)
  }


  private def generateInitWeights(): Vector = {
    (k0, k1) match {
      case (true, true) =>
        Vectors.dense(Array.fill(numFeatures * k2 + numFeatures)(Random.nextGaussian() * initStd + initMean) ++ Array.fill(1)(0.0))

      case (true, false) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++  Array.fill(1)(0.0))

      case (false, true) =>
        Vectors.dense(Array.fill(numFeatures * k2 + numFeatures)(Random.nextGaussian() * initStd + initMean))

      case (false, false) =>
        Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean))
    }
  }


  private def createModel(weights: Vector): FMModelV3 = {

    val values = weights.toArray

    val v = new DenseMatrix(k2, numFeatures, values.slice(0, numFeatures * k2))

    val w = if (k1) Some(Vectors.dense(values.slice(numFeatures * k2, numFeatures * k2 + numFeatures))) else None

    val w0 = if (k0) values.last else 0.0

    new FMModelV3(task, v, w, w0, minLabel, maxLabel)
  }
}
