package org.apache.spark.ml.made

import scala.util.control.Breaks._
import breeze.linalg
import breeze.linalg.{*, DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector, Matrix => BreezeMatrix, Vector => BreezeVector}
import breeze.stats.mean
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.{
	DefaultParamsReader, 
	DefaultParamsWritable, 
	DefaultParamsWriter, 
	Identifiable, 
	MLReadable, 
	MLReader,
	MLWritable,
	MLWriter,
	MetadataUtils
}
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.made.LinearRegressionModel.LinearRegressionModelWriter
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.mllib
import scala.collection.mutable.ArrayBuffer

trait HasLR extends Params {
  final val lr: DoubleParam = new DoubleParam(this, "lr", "gd learning rate", ParamValidators.gtEq(0))
  final def getLR: Double = $(lr)
}

trait LinearRegressionParams 
	extends PredictorParams 
	with HasLR 
	with HasMaxIter 
	with HasTol {
  		override protected def validateAndTransformSchema(
                                      schema: StructType,
                                      fitting: Boolean,
                                      featuresDataType: DataType): StructType = {
    		super.validateAndTransformSchema(schema, fitting, featuresDataType)
 	}

  setDefault(lr -> 1e-4)
  setDefault(maxIter -> 1000)
  setDefault(tol -> 1e-5)
}

class LinearRegressionGD(override val uid: String)
  extends Regressor[Vector, LinearRegressionGD, LinearRegressionModel]
    with LinearRegressionParams 
    with DefaultParamsWritable 
    with Logging {

  def this() = this(Identifiable.randomUID("linRegGD"))
  def setLR(value: Double): this.type = set(lr, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setTol(value: Double): this.type = set(tol, value)
  override def copy(extra: ParamMap): LinearRegressionGD = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

    implicit val vectorEncoder : Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder : Encoder[Double] = ExpressionEncoder()

    var w: BreezeDenseVector[Double] =
    			BreezeDenseVector.rand[Double](numFeatures)
    var b: Double = 1.0
    var error: Double = Double.MaxValue

    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    breakable {for (i <- 1 to getMaxIter) {
      val (coefficientsSummary, interceptSummary) = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val coefficientsSummarizer = new MultivariateOnlineSummarizer()
        val interceptSummarizer = new MultivariateOnlineSummarizer()

        data.grouped(1000).foreach((r: Seq[(Vector, Double)]) => {
          val (x_, y_) = r.map(x => (
            x._1.toArray.to[ArrayBuffer], Array(x._2).to[ArrayBuffer]
          )).reduce((x, y) => {
            (x._1 ++ y._1, x._2 ++ y._2)
          })

          // Create breeze matrix and dense vector
          val x__ = x_.toArray
          val y__ = y_.toArray
          val X = BreezeDenseMatrix.create(x__.size / numFeatures, numFeatures, x__, 0, numFeatures, true)
          val Y = BreezeDenseVector(y__)

          var output = (X * w) + b

          val residuals = Y - output

          val c: BreezeDenseMatrix[Double] = X(::, *) * residuals

          coefficientsSummarizer.add(mllib.linalg.Vectors.fromBreeze(mean(c(::, *)).t))
          interceptSummarizer.add(mllib.linalg.Vectors.dense(mean(residuals)))
        })

        Iterator((coefficientsSummarizer, interceptSummarizer))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })

      error = interceptSummary.mean(0)
      if (error.abs < getTol)
        break

      var dCoeff: BreezeDenseVector[Double] = coefficientsSummary.mean.asBreeze.toDenseVector
      dCoeff :*= (-2.0) * getLR
      w -= dCoeff

      var dInter = (-2.0) * getLR * error
      b -= dInter
    } }

    val lrModel = copyValues(new LinearRegressionModel(uid, new DenseVector(w.toArray), b))
    lrModel
} }



class LinearRegressionModel private[made](
                                   override val uid: String,
                                   val w: Vector,
                                   val b: Double)
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams with MLWritable {

  val brzCoefficients: BreezeVector[Double] = w.asBreeze

  private[made] def this(w: Vector, b: Double) = this(Identifiable.randomUID("linRegGD"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(w, b))

  override def write: MLWriter = new LinearRegressionModelWriter(this)

  override def predict(features: Vector): Double = {
    (features.asBreeze dot brzCoefficients) + b
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  class LinearRegressionModelWriter(instance: LinearRegressionModel) extends MLWriter {
    private case class Data(b: Double, w: Vector)

    override def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)

      val data = Data(instance.b, instance.w)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {
    private val className = classOf[LinearRegressionModel].getName

    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(b: Double, w: Vector) = data.select("b", "w").head()
      val model = new LinearRegressionModel(metadata.uid, w, b)

      metadata.getAndSetParams(model)
      model
    }
  }
  override def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader
  override def load(path: String): LinearRegressionModel = super.load(path)
}
