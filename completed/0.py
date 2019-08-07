import re, sys
from pyspark import SparkConf, SparkContext
sc = SparkContext(conf=SparkConf())

lines = sc.textFile('/home/u/Desktop/hw0-bundle/pg100.txt')
words = lines.flatMap(lambda l: re.findall(r"[\w'-]+", l))
words = words.filter(lambda w: w[0].isalpha())
pairs = words.map(lambda w: (w[0].lower(), 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2) # pair[0] is treated as key
sort = counts.sortBy(lambda w: w[1], False)
sort.saveAsTextFile('./0')

sc.stop()
