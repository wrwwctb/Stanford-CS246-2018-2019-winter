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
edges_uv = lines.map(lambda line: list(map(int, line.split())))
edges_vu = lines.map(lambda line: list(map(int, line.split()[::-1])))

def aggregateUV(st, v):
    st.add(v)
    return st

def aggregateUU(st1, st2):
    return st1 | st2

u_outNeigs = edges_uv.aggregateByKey(set(), aggregateUV, aggregateUU)
v_inNeigs  = edges_vu.aggregateByKey(set(), aggregateUV, aggregateUU)
# u_outDegs = u_outNeigs.map(lambda u_st: (u_st[0], len(u_st[1])))

M_uv = {}
M_vu = {}
for u, outNeigs in u_outNeigs.collect():
    M_uv[u] = outNeigs

for v, inNeigs in v_inNeigs.collect():
    M_vu[v] = inNeigs

u_outNeigs = sc.broadcast(M_uv)
v_inNeigs  = sc.broadcast(M_vu)


numIters = 40

def LTh(h):
    LT = u_outNeigs.value
    new_a = [0] * n
    for j, hj in enumerate(h, 1): # 0/1 indexing
        if hj == 0:
            continue
        for i in LT[j]:
            new_a[i - 1] += hj # 0/1 indexing
    return new_a

def L_a(a):
    L  = v_inNeigs.value
    new_h = [0] * n
    for j, aj in enumerate(a, 1): # 0/1 indexing
        if aj == 0:
            continue
        for i in L[j]:
            new_h[i - 1] += aj # 0/1 indexing
    return new_h

h = [1] * n


for _ in range(numIters):
    a = LTh(h)
    a = [v / max(a) for v in a]
    h = L_a(a)
    h = [v / max(h) for v in h]

h_id = zip(h, range(1, n + 1))
a_id = zip(a, range(1, n + 1))

h_id = sorted(h_id, reverse=True)
a_id = sorted(a_id, reverse=True)

print(h_id[:5])
print(h_id[-5:])
print(a_id[:5])
print(a_id[-5:])






