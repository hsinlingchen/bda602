from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

rolling_sql = """SELECT bg1.batter
                     , bg1.game_id
                     , bg1.local_date
                     , SUM(bg2.Hit)/SUM(bg2.atBat) AS rolling_ba
                     FROM batter_game bg1
                     JOIN batter_game bg2
                     ON bg1.batter = bg2.batter
                     AND bg2.local_date
                     BETWEEN DATE_SUB(bg1.local_date, 100)
                     AND DATE_SUB(bg1.local_date, 1)
                     WHERE bg2.atBat > 0
                     GROUP BY bg1.batter, bg1.game_id, bg1.local_date
                     ORDER BY bg1.batter, bg1.game_id ASC
                """


class Rolling100Transform(Transformer):
    @keyword_only
    def __init__(self):
        super(Rolling100Transform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self):
        spark = SparkSession.builder.master("local[*]").getOrCreate()
        dataset = spark.sql(rolling_sql)
        return dataset
