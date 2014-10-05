import numpy as np
import numpy.matlib as mp
import csv
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt
from tabulate import tabulate

# Make the graphs a bit prettier, and bigger
# pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

np.set_printoptions(threshold=np.nan)

parser = ap.ArgumentParser(description="NFL Pagerank team rankings")
parser.add_argument('--in', action='store', dest='infile')
parser.add_argument('--out', action='store', dest='outfile')
parser.add_argument('--d', action='store', dest='d', type=float)
options = parser.parse_args()

N = 32.
ADJ = mp.zeros((N,N))
Elo = mp.zeros((N,1))
teams = {}
teamsA = []
wins = np.zeros(N)
losses = np.zeros(N)
fh = open(options.infile, 'rb')
reader = csv.reader(fh)
count = 0
for row in reader:
    if not teams.has_key(row[0]):
        teamsA.append(row[0])
        teams[row[0]] = len(teamsA) - 1
    if not teams.has_key(row[1]):
        teamsA.append(row[1])
        teams[row[1]] = len(teamsA) - 1
    wins[teams[row[0]]] += 1
    losses[teams[row[1]]] += 1
    diff = float(row[2]) - float(row[3])
    ADJ[teams[row[0]], teams[row[1]]] += diff
    ADJ[teams[row[1]], teams[row[0]]] += -diff

ADJ = pd.DataFrame( ADJ, index=teamsA, columns=teamsA)
Elo = pd.DataFrame( Elo, index=teamsA, columns=['Elo'])
K = 1
norms = []
for l in range(0,20):
    norms.append(np.linalg.norm(Elo))
    # print Elo.sort(columns='Elo', ascending=False)
    tens = np.mat(np.zeros((N,1))) + 10
    Q = np.power(tens, Elo)
    # print "Q: \n", Q
    E = mp.zeros((N,N))
    E = pd.DataFrame(E, index=teamsA, columns=teamsA)
    ADJ = np.array(ADJ)
    table = []
    for i in range(0, int(N)):
        for j in range(0, int(N)):
            # print "i: ", i, "Q[i]: ", Q[i]
            # print "j: ", j, "Q[j]: ", Q[j]
            if( abs(ADJ[i,j]) ):
                E.iat[i,j] = 20 * (Q[i]/(Q[j] + Q[i]) - 0.5 ) * 2.
                table.append( [teamsA[i], Elo.iat[i,0],
                               teamsA[j], Elo.iat[j,0],
                               ADJ[i,j], E.iloc[i,j],
                               ADJ[i,j] - E.iloc[i,j]])
    print tabulate(table, floatfmt='.2f')
    # E = np.multiply(abs(ADJ), E)
    # E=pd.DataFrame(E, index=teamsA, columns=teamsA)
    # print "E: ", E
    Elo = Elo + (K * (ADJ - np.array(E))) * mp.ones((N,1))
    norm = np.linalg.norm(Elo)
    # Elo = Elo/norm
    norms.append(np.linalg.norm(Elo))
    # Elo = Elo /
plt.plot(norms)
sorted_Elo = Elo.sort(columns='Elo', ascending=False)
sorted_Elo.plot()
print sorted_Elo
plt.show()
# import sys; sys.exit()

# print "min: ", ADJ.min()
# ADJ = ADJ - ADJ.min()
"""
L = ADJ
d = options.d
D = np.mat(np.zeros((N,1))) + (1. - d)/N

df = pd.DataFrame( np.array(Elo).T[0], index=teamsA, columns=['PageRank'])
df['W'] = wins
df['L'] = losses
df[['W', 'L']] = df[['W', 'L']].astype(int)
df['PageRank'] *= 100
dfs = df.sort(columns='PageRank', ascending=False)
print dfs
dfs.to_csv(options.outfile, sep='|',
           float_format='%2.2f', index_label='Team')
# Elo = np.array(Elo).T[0]
# zipped = zip(teamsA, Elo)
# sortd = sorted(zipped, key=lambda x: x[1], reverse=True )
# print pandas.DataFrame(sortd)
# print "sorted: \n", sortd
# flattened = [item for sublist in Elo for item in sublist]
# print "flattened: ", flattened
"""
