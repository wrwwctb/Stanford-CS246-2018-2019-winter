from pyspark import SparkConf, SparkContext
sc = SparkContext(conf=SparkConf())

#dataPath = './data/graph-small.txt' # node # 1 ~ 100
dataPath = './data/graph-full.txt' # node # 1 ~ 1000

# check min and max node #
mi = float('inf')
Mx = -float('inf')
with open(dataPath) as f:
    for line in f:
        a, b = map(int, line.split())
        mi = min(mi, a, b)
        Mx = max(Mx, a, b)

n = Mx

lines = sc.textFile(dataPath)
edges = lines.map(lambda line: list(map(int, line.split())))

def aggregateUV(st, v):
    st.add(v)
    return st

def aggregateUU(st1, st2):
    return st1 | st2

u_outNeigs = edges.aggregateByKey(set(), aggregateUV, aggregateUU)
# u_outDegs = u_outNeigs.map(lambda u_st: (u_st[0], len(u_st[1])))

M = {}
for u, outNeigs in u_outNeigs.collect():
    M[u] = outNeigs

u_outNeigs = sc.broadcast(M)

beta = .8
one_beta_n = (1 - beta) / n
numIters = 40

def Mr(r):
    M = u_outNeigs.value
    new = [0] * n
    for j, rj in enumerate(r, 1): # 0/1 indexing
        if rj == 0:
            continue
        # assert deg != 0
        inv = rj / len(M[j])
        for i in M[j]:
            new[i - 1] += inv # 0/1 indexing
    return new


r = [1 / n] * n


for _ in range(numIters):
    r = [one_beta_n + beta * val for val in Mr(r)]

r_id = zip(r, range(1, n + 1))
r_id = sorted(r_id, reverse=True)

print(r_id[:5])
print(r_id[-5:])







