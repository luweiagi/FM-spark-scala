package impl

import conf.FMConfigUtil
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
//import org.apache.spark.mllib.regression._
import model.FMV3._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

object TestFMV3 {

  def main(args: Array[String]): Unit = {

    println("==================================变量初始化==========================================================")
    // spark参数设置
    val spark = SparkSession
      .builder()
      //.config(sparkConf)
      .appName("TestFM")
      .config("spark.driver.maxResultSize", "10g")
      .config("spark.rpc.message.maxSize", "512")
      .enableHiveSupport()
      .getOrCreate()

    spark.sparkContext.hadoopConfiguration.set("mapred.output.compress", "false")
    val sc = spark.sparkContext

    spark.sqlContext.setConf("hive.exec.dynamic.partition", "true")
    spark.sqlContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")
    val sqlcontext = spark.sqlContext

    if (args.length < 4) {
      System.out.println("parameter error!")
      System.exit(-1)
    }

    /* 参数输入项：
    *  (0):partition_date 日期
    *  (1):numIterations  训练轮数
    *  (2):opt            优化方法
    *  (3):stepSize       学习率
    * */

    val partition_date = args(0)
    println("partition_date = " + partition_date)

    // model param
    val task = 1  // 回归任务0，分类任务1
    val numIterations = Math.min(args(1).toDouble.toInt, 100)  // 训练轮数
    val opt = args(2).toLowerCase  // 优化器，目前支持SGD、Adam和RMSProp
    val stepSize = args(3).toDouble  // 学习率
    val dim = (true, true, 8)  // 0：零阶项w0是否存在，1：一阶项w1-wn是否存在，2：二阶交叉项w1-wn的embedding维度
    val regParam = (0.0, 0.0, 0.001)  // 0：零阶项w0的l2正则项，1：一阶项w1-wn的l2正则项，2：二阶交叉项w1-wn的l2正则项
    val initStd = 0.1  // 初始化embedding的标准差

    opt.toLowerCase match {
      case "sgd" => ;
      case "rmsprop" => ;
      case "adam" => ;
      case _ => println("opt not surport!" + opt); System.exit(-1)
    }

    println("task = " + task)
    println("epoch = " + numIterations)
    println("opt: " + opt)
    println("学习率 = " + stepSize)
    println("零阶项w0是否存在: " + dim._1)
    println("一阶项w1-wn是否存在: " + dim._2)
    println("二阶交叉项w1-wn的embedding维度: " + dim._3)
    println("正则项:")
    println("\t零阶项w0的l2正则项: " + regParam._1)
    println("\t一阶项w1-wn的l2正则项: " + regParam._2)
    println("\t二阶交叉项w1-wn的l2正则项: " + regParam._3)
    println("初始化embedding的标准差: " + initStd)


    println("==================================下载数据==========================================================")

    val conf = FMConfigUtil.getDefaultConfig

    val numPartition = 256

    // libsvm训练数据地址
    val train_libsvm_data_path = conf.getString("train_libsvm_data_path_for_train")
    val valid_libsvm_data_path = conf.getString("valid_libsvm_data_path_for_train")
    val feature_embedding_path = conf.getString("feature_embedding_path")

    println("load train data")
    // 注意：loadLibSVMFile会给每个index减1重新变为index从0开始的，所以数据格式需要从1开始（即标准的libsvm格式）
    val trainData = MLUtils.loadLibSVMFile(sc, train_libsvm_data_path).map(
      x =>
        LabeledPoint(if(x.label > Double.MinPositiveValue) 1.0 else 0.0, x.features)
    ).repartition(numPartition).persist(StorageLevel.MEMORY_AND_DISK)
    println("load valid data")
    // 注意：loadLibSVMFile会给每个index减1重新变为index从0开始的，所以数据格式需要从1开始（即标准的libsvm格式）
    val validData = MLUtils.loadLibSVMFile(sc, valid_libsvm_data_path).map(
      x =>
        LabeledPoint(if(x.label > Double.MinPositiveValue) 1.0 else 0.0, x.features)
    ).repartition(numPartition).persist(StorageLevel.MEMORY_AND_DISK)

    println("trainData examples:")
    trainData.take(3).foreach(x => println("\t" + x))
    println("validData examples:")
    validData.take(3).foreach(x => println("\t" + x))


    println("===========================描述训练数据和验证数据的信息================================================")

    // x._1 is each feature max index, x._2 is label(0/1)
    val trainInfo = trainData.map { x =>
      (x.features.toSparse.indices.lastOption.getOrElse(0), if(x.label > Double.MinPositiveValue) 1 else 0)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val validInfo = validData.map { x =>
      (x.features.toSparse.indices.lastOption.getOrElse(0), if(x.label > Double.MinPositiveValue) 1 else 0)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val numFeatures = trainInfo.map(x => x._1).reduce(math.max)// + 1

    val (numPosTrain, numTotalTrain) = trainInfo.map(x => x._2).treeAggregate((0d, 0))(
      (acc, label) => (acc._1 + label, acc._2 + 1),
      (par1, par2) => (par1._1 + par2._1, par1._2 + par2._2)
    )

    val (numPosValid, numTotalValid) = validInfo.map(x => x._2).treeAggregate((0d, 0))(
      (acc, label) => (acc._1 + label, acc._2 + 1),
      (par1, par2) => (par1._1 + par2._1, par1._2 + par2._2)
    )

    trainInfo.unpersist()
    validInfo.unpersist()

    println("numFeatures = " + numFeatures)
    println("numPartition = " + numPartition)
    println("train:")
    println("\tnumTotalTrain = " + numTotalTrain)
    println("\tposRatio = " + (numPosTrain.toDouble / numTotalTrain.toDouble * 100.0).formatted("%.2f") + "%")
    println("\tpos : neg = 1 : " + ((numTotalTrain - numPosTrain) / numPosTrain).formatted("%.1f"))
    println("valid:")
    println("\tnumTotalValid = " + numTotalValid)
    println("\tposRatio = " + (numPosValid.toDouble / numTotalValid.toDouble * 100.0).formatted("%.2f") + "%")
    println("\tpos : neg = 1 : " + ((numTotalValid - numPosValid) / numPosValid).formatted("%.1f"))


    println("==================================训练模型==========================================================")

    val fm_model: FMModelV3 = FMWithSGDV3.train(
      trainData,  // 训练数据
      validData,  // 验证数据
      task = task,  // 回归任务0，分类任务1
      numIterations = numIterations,  // 训练轮数
      opt = opt.toLowerCase,  // 优化器，目前支持sgd、adam、RMSProp
      stepSize = stepSize,  // 学习率
      dim = dim,  // 0：零阶项w0是否存在，1：一阶项w1-wn是否存在，2：二阶交叉项w1-wn的embedding维度
      regParam = regParam,  // 0：零阶项w0的l2正则项，1：一阶项w1-wn的l2正则项，2：二阶交叉项w1-wn的l2正则项
      initStd = initStd,  // 初始化embedding的标准差
      numPartition = numPartition
    )


    println("==================================模型预测==========================================================")

    // 训练数据 (正样本的预测得分加和, 正样本的总数, 负样本的预测得分加和, 负样本的总数)
    val scoresPredictTrain: (Double, Int, Double, Int) = trainData.map(
      _.label
    ).zip(
      fm_model.predict(trainData.map(_.features))
    ).treeAggregate((0d, 0, 0d, 0))(
      (acc, value) => (
        if(value._1 > 0.0) acc._1 + value._2 else acc._1,
        if(value._1 > 0.0) acc._2 + 1 else acc._2,
        if(value._1 < 0.0) acc._3 + value._2 else acc._3,
        if(value._1 < 0.0) acc._4 + 1 else acc._4
      ),
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2, acc1._3 + acc2._3, acc1._4 + acc2._4)
    )

    val posPredictTrain = scoresPredictTrain._1 / scoresPredictTrain._2.toDouble
    val negPredictTrain = scoresPredictTrain._3 / scoresPredictTrain._4.toDouble
    println("train positive samples is predicted as: " + posPredictTrain.formatted("%.4f"))
    println("train negtive  samples is predicted as: " + negPredictTrain.formatted("%.4f"))

    // 验证数据 (正样本的预测得分加和, 正样本的总数, 负样本的预测得分加和, 负样本的总数)
    val scoresPredictValid: (Double, Int, Double, Int) = validData.map(
      _.label
    ).zip(
      fm_model.predict(validData.map(_.features))
    ).treeAggregate((0d, 0, 0d, 0))(
      (acc, value) => (
        if(value._1 > 0.0) acc._1 + value._2 else acc._1,
        if(value._1 > 0.0) acc._2 + 1 else acc._2,
        if(value._1 < 0.0) acc._3 + value._2 else acc._3,
        if(value._1 < 0.0) acc._4 + 1 else acc._4
      ),
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2, acc1._3 + acc2._3, acc1._4 + acc2._4)
    )

    val posPredictValid = scoresPredictValid._1 / scoresPredictValid._2.toDouble
    val negPredictValid = scoresPredictValid._3 / scoresPredictValid._4.toDouble
    println("valid positive samples is predicted as: " + posPredictValid.formatted("%.4f"))
    println("valid negtive  samples is predicted as: " + negPredictValid.formatted("%.4f"))


    println("==================================保存模型参数=======================================================")

    val v = fm_model.getV
    val w = fm_model.getW
    val w0 = fm_model.getW0

    if (w.isEmpty) {
      println("本模型设定为不包含一阶项！")
    } else {
      println("一阶项的特征w数量为(不包含w0): " + w.getOrElse(Vector(0)).asInstanceOf[Vector[Double]].size)
    }

    println("二阶交叉项的特征v数量为: " + v.numCols + ", 二阶交叉项的特征v的维度为: " + v.numRows)
    if (w.size != v.numCols) println("警告：二阶交叉项的特征v数量不等于一阶项的特征w数量！请检查模型！！！")

    val modelBuffer = new ArrayBuffer[String]()
    modelBuffer.+=("w_" + 0 + " " + w0)
    var i = 1
    w.foreach(x => {modelBuffer.+=("w_" + i + " " + x.toString); i+=1})
    i = 1
    v.colIter.foreach(x => {modelBuffer.+=("v_" + i + " " + x.toArray.mkString(" ")); i+=1})

    // 预先删除输出文件夹的内容以及最后一层的文件夹
    val feature_embedding_path_fs = new Path(feature_embedding_path).getFileSystem(sc.hadoopConfiguration)
    try {
      feature_embedding_path_fs.delete(new Path(feature_embedding_path), true)
    } catch {
      case e: Exception => println("delete hdfs file error! " + e)
    }

    sc.parallelize(modelBuffer,30).saveAsTextFile(feature_embedding_path)


    println("==================================end==============================================================")

    sc.stop()
  }

}

