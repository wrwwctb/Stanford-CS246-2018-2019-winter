import re, sys
from pyspark import SparkConf, SparkContext
sc = SparkContext(conf=SparkConf())

lines = sc.textFile('/home/u/Desktop/hw1-bundle/q1/data/soc-LiveJournal1Adj.txt')

def lines2cands(l):
    uvs = re.split(r'[,\t]', l.strip()) # 'u' 'a' 'b' 'c' 'd'
    uvs = list(map(int, uvs)) # u a b c d
    u = uvs[0]
    vs = sorted(uvs[1:])
    # each cand pair has score 1
    ret = []
    for i in range(len(vs)):
        for j in range(i):
            ret.append(((vs[j], vs[i]), 1))
    # if already connected, score -inf
    minf = -float('inf')
    for v in vs:
        if u < v:
            ret.append(((u, v), minf))
        else:
            ret.append(((v, u), minf))
    # returns [] if no friend. flatMap ignores []
    return ret

cands = lines.flatMap(lines2cands)

cnt = cands.reduceByKey(lambda x, y: x + y) # ((a b) num)
cnt = cnt.filter(lambda kv: kv[1] > 0) # filter connected
# up to here, those w/o 2nd deg friends are excluded

def cnt2suggest(uv_num):
    uv, num = uv_num
    u, v = uv
    return (u, (num, v)), (v, (num, u))

suggest = cnt.flatMap(cnt2suggest) # (a, (num b)) (b, (num a))
suggest = suggest.groupByKey() # (a, ((num b) (num c)) )

def processSuggest(suggest):
    u, vs = suggest
    # sort by number of common friends
    vs = sorted(vs, key=lambda v: (-v[0], v[1]))
    vs = vs[:10]
    # remove numbers. leave friend ids
    vs = list(zip(*vs))[1]
    return u, vs

suggest = suggest.map(processSuggest)

suggest = suggest.sortBy(lambda u_vs: u_vs[0])

suggest.saveAsTextFile('./1_1')

sc.stop()



