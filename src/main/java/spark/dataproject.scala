package spark

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession,SaveMode}
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col

object dataproject {

  def main(args: Array[String]):Unit= {
    val spark = SparkSession
      .builder
      .appName("city")
      .config("spark.local.dir", "E:\\Scala—project\\temp")
      //.config("spark.hadoop.fs.defaultFS", "hdfs://master:8020")
      .master("local[2]")
      .getOrCreate()



    val jdbcUrl = "jdbc:mysql://localhost:3306/spark_data"
    val connectionProperties = new java.util.Properties()
    connectionProperties.put("user", "root")
    connectionProperties.put("password", "225724")
    connectionProperties.put("driver", "com.mysql.cj.jdbc.Driver")

    val schema = new StructType(Array(
      StructField("province", DataTypes.StringType),
      StructField("city", DataTypes.StringType),
      StructField("administrative_level", DataTypes.StringType),
      StructField("region", DataTypes.StringType),
      StructField("per_capita_disposable_income", DataTypes.DoubleType),
      StructField("housing_price_to_income_ratio", DataTypes.DoubleType),
      StructField("educational_satisfaction", DataTypes.DoubleType),
      StructField("medical_resource_index", DataTypes.DoubleType),
      StructField("annual_average_PM2.5", DataTypes.DoubleType),
      StructField("park_green_area", DataTypes.DoubleType),
      StructField("pension_insurance_coverage_rate", DataTypes.DoubleType),
      StructField("police_forces_per_10000_people", DataTypes.DoubleType),
      StructField("commuting_time", DataTypes.DoubleType),
      StructField("happiness_Index", DataTypes.DoubleType)))

    val inputpath = "D:\\22572\\桌面\\期末实训\\inputdata\\updated_city_happiness.csv"
    //val inputpath = "hdfs://master:8020/user/hadoop/data/updated_city_happiness.csv"
    //val outputpath = "D:\\22572\\桌面\\期末实训\\outputdata"
    val df = spark
      .read
      .format("csv")
      .schema(schema)
      .option("header", "true")
      .option("delimiter",",")
      .csv(inputpath)

    // 数据清洗
    val cleanedDF = df.filter(
      col("city").isNotNull &&
        col("happiness_index").isNotNull &&
        col("per_capita_disposable_income").isNotNull&&
        col("`Annual_average_PM2.5`") > 0.1
    ).cache() // 缓存清洗后的数据


    // 处理缺失值（更精细的处理）
    val finalDF = cleanedDF.na.fill(0, Seq(
      "educational_satisfaction",
      "medical_resource_index",
      "park_green_area",
      "pension_insurance_coverage_rate",
      "police_forces_per_10000_people"
    )).na.fill("Unknown", Seq(
      "province",
      "city",
      "administrative_level",
      "region"
    ))

    println("清洗后数据示例:")
    finalDF.show(5, truncate = false)
    println(s"有效记录数: ${finalDF.count()}")


    //.load("hdfs://master:8020/")
    //df.show(numRows=10)
    //创建虚拟视图
    //df.createTempView("city")
    /*val scoreall = spark.sql("select * from city offset 1")
    scoreall.show
    scoreall
      .write
      .format("csv")
      .option("header", "true") // 是否包含列名
      .option("encoding", "UTF-8")
      .mode("overwrite") // 写入模式：overwrite, append, ignore, error
      .save(outputpath)*/

    //数据处理过程
    val GDP = df.orderBy(col("Per_capita_disposable_income").desc).limit(10)
    GDP.show()

    GDP.write
    .mode(SaveMode.Overwrite) // 可以是Append, Overwrite, ErrorIfExists, Ignore
    .option("truncate", "false")
    .jdbc(jdbcUrl, "city", connectionProperties)
    println("GDP数据成功写入MySQL数据库！")


    val sorteddf = finalDF.orderBy(col("`Annual_average_PM2.5`")).filter("`Annual_average_PM2.5` < 50")
    sorteddf.show()

    sorteddf.write
      .mode(SaveMode.Overwrite) // 可以是Append, Overwrite, ErrorIfExists, Ignore
      .option("truncate", "false")
      .jdbc(jdbcUrl, "city2", connectionProperties)
    println("sorteddf数据成功写入MySQL数据库！")

    val housedf = finalDF.orderBy(col("housing_price_to_income_ratio")).filter("housing_price_to_income_ratio < 6")
    housedf.show()

    housedf.write
      .mode(SaveMode.Overwrite) // 可以是Append, Overwrite, ErrorIfExists, Ignore
      .option("truncate", "false")
      .jdbc(jdbcUrl, "city3", connectionProperties)
    println("housedf数据成功写入MySQL数据库！")

    // 描述性统计
    println("关键指标描述性统计:")
    val key_value_describe_count = finalDF.describe(
      "per_capita_disposable_income",
      "housing_price_to_income_ratio",
      "happiness_index"
    )
    key_value_describe_count.show()
    // 2. 确认MySQL连接
    try {
      val conn = java.sql.DriverManager.getConnection(jdbcUrl, connectionProperties)
      println("成功连接到MySQL")

      // 检查表是否存在
      val dbmd = conn.getMetaData()
      val tables = dbmd.getTables(null, null, "key_value_describe_count", null)
      if(tables.next()) {
        println("表 key_value_describe_count 存在")
      } else {
        println("表 key_value_describe_count 不存在，将创建")
        val stmt = conn.createStatement()
        stmt.execute("""
      CREATE TABLE key_value_describe_count (
        per_capita_disposable_income DOUBLE,
        housing_price_to_income_ratio DOUBLE,
        happiness_index DOUBLE
      )
    """)
        stmt.close()
      }
      conn.close()
    } catch {
      case e: Exception =>
        println("MySQL连接/表检查失败: " + e.getMessage)
        return
    }

    // 3. 执行写入并确认
    println("开始写入MySQL...")
    key_value_describe_count.write
      .mode(SaveMode.Overwrite)
      .option("truncate", "false")
      .option("batchsize", 10000)
      .jdbc(jdbcUrl, "key_value_describe_count", connectionProperties)
    println("写入操作完成")

    // 按行政级别分组统计
    println("按行政级别分组统计:")
    val administrative_level_count = finalDF.groupBy("administrative_level")
      .agg(
        avg("happiness_index").as("avg_happiness"),
        avg("per_capita_disposable_income").as("avg_income"),
        avg("housing_price_to_income_ratio").as("avg_house_price_ratio"),
        count("*").as("city_count")
      )
      .orderBy(desc("avg_happiness"))
    administrative_level_count.show()
    // 2. 确认MySQL连接
    try {
      val conn = java.sql.DriverManager.getConnection(jdbcUrl, connectionProperties)
      println("成功连接到MySQL")

      // 检查表是否存在
      val dbmd = conn.getMetaData()
      val tables = dbmd.getTables(null, null, "administrative_level_count", null)
      if(tables.next()) {
        println("表 administrative_level_count 存在")
      } else {
        println("表 administrative_level_count 不存在，将创建")
        val stmt = conn.createStatement()
        stmt.execute("""
      CREATE TABLE administrative_level_count (
        avg_happiness DOUBLE,
        avg_income DOUBLE,
        avg_house_price_ratio DOUBLE,
        city_count DOUBLE
      )
    """)
        stmt.close()
      }
      conn.close()
    } catch {
      case e: Exception =>
        println("MySQL连接/表检查失败: " + e.getMessage)
        return
    }

    // 3. 执行写入并确认
    println("开始写入MySQL...")
    administrative_level_count.write
      .mode(SaveMode.Overwrite)
      .option("truncate", "false")
      .option("batchsize", 10000)
      .jdbc(jdbcUrl, "administrative_level_count", connectionProperties)
    println("写入操作完成")

    // 按地区分组统计
    println("按地区分组统计:")
    val region_count = finalDF.groupBy("region")
      .agg(
        count("*").as("city_count"),
        avg("happiness_index").as("avg_happiness"),
        avg("`annual_average_PM2.5`").as("avg_pm25"),
        avg("park_green_area").as("avg_green_area")
      )
      .orderBy(desc("avg_happiness"))
    region_count.show()

    // 2. 确认MySQL连接
    try {
      val conn = java.sql.DriverManager.getConnection(jdbcUrl, connectionProperties)
      println("成功连接到MySQL")

      // 检查表是否存在
      val dbmd = conn.getMetaData()
      val tables = dbmd.getTables(null, null, "region_count", null)
      if(tables.next()) {
        println("表 region_count 存在")
      } else {
        println("表 region_count 不存在，将创建")
        val stmt = conn.createStatement()
        stmt.execute("""
      CREATE TABLE region_count (
        avg_happiness DOUBLE,
        avg_income DOUBLE,
        avg_house_price_ratio DOUBLE,
        city_count DOUBLE
      )
    """)
        stmt.close()
      }
      conn.close()
    } catch {
      case e: Exception =>
        println("MySQL连接/表检查失败: " + e.getMessage)
        return
    }

    // 3. 执行写入并确认
    println("开始写入MySQL...")
    region_count.write
      .mode(SaveMode.Overwrite)
      .option("truncate", "false")
      .option("batchsize", 10000)
      .jdbc(jdbcUrl, "region_count", connectionProperties)
    println("写入操作完成")

    // 计算幸福指数与其他指标的相关性
    println("幸福指数与其他指标的相关性:")
    val corrDF = finalDF.select(
      corr("happiness_index", "per_capita_disposable_income").as("corr_income"),
      corr("happiness_index", "educational_satisfaction").as("corr_education"),
      corr("happiness_index", "medical_resource_index").as("corr_medical"),
      corr("happiness_index", "park_green_area").as("corr_green_area"),
      corr("happiness_index", "`annual_average_PM2.5`").as("corr_pm25"),
      corr("happiness_index", "commuting_time").as("corr_commute")
    )
    corrDF.show()

    corrDF.write.mode(SaveMode.Append)// 可以是Append, Overwrite, ErrorIfExists, Ignore
      .option("truncate", "false")
      .jdbc(jdbcUrl, "xiangguanxingxishu", connectionProperties)
    println("数据成功写入MySQL数据库！")
    println("=============================================")


    // 1. 确认数据存在
    println("相关性数据:")
    corrDF.show()

    // 2. 确认MySQL连接
    try {
      val conn = java.sql.DriverManager.getConnection(jdbcUrl, connectionProperties)
      println("成功连接到MySQL")

      // 检查表是否存在
      val dbmd2 = conn.getMetaData()
      val tables = dbmd2.getTables(null, null, "xiangguanxingxishu", null)
      if(tables.next()) {
        println("表 xiangguanxingxishu 存在")
      } else {
        println("表 xiangguanxingxishu 不存在，将创建")
        val stmt = conn.createStatement()
        stmt.execute("""
      CREATE TABLE xiangguanxingxishu (
        corr_income DOUBLE,
        corr_education DOUBLE,
        corr_medical DOUBLE,
        corr_green_area DOUBLE,
        corr_pm25 DOUBLE,
        corr_commute DOUBLE
      )
    """)
        stmt.close()
      }
      conn.close()
    } catch {
      case e: Exception =>
        println("MySQL连接/表检查失败: " + e.getMessage)
        return
    }

    // 3. 执行写入并确认
    println("开始写入MySQL...")
    corrDF.write.mode(SaveMode.Overwrite)
      .option("truncate", "false")
      .option("batchsize", 10000)
      .jdbc(jdbcUrl, "xiangguanxingxishu", connectionProperties)
    println("写入操作完成")

  }
}
