import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from transformer import Rolling100Transform

# Setup Spark
spark = SparkSession.builder.master("local[*]").getOrCreate()

database = "baseball"
user = "root"
password = "root"  # pragma: allowlist secret
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

bg_query = """SELECT bc.batter
                     , bc.game_id
                     , DATE(g.local_date) AS local_date
                     ,bc.atBat
                     ,bc.Hit
                FROM batter_counts bc
                JOIN game g
                ON bc.game_id = g.game_id
                ORDER BY bc.batter, bc.game_id ASC"""

df_bg = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", bg_query)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)

# Store the Dataframe
df_bg.createOrReplaceTempView("batter_game")
df_bg.persist(StorageLevel.DISK_ONLY)


def main():
    # show the join table selected from mariadb
    print("First 25 Rows of Intermediary Join Table From mariadb:")
    df_bg.show(25)
    roll_100 = Rolling100Transform()
    roll_100_dataframe = roll_100._transform()
    # show table of the rolling 100 result
    print("First 25 Rows of Rolling 100 Day Stats with Transformer:")
    roll_100_dataframe.show(25)
    return


if __name__ == "__main__":
    sys.exit(main())
