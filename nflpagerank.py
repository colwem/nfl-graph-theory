import numpy as np
import csv
import pandas as pd
import argparse as ap

parser = ap.ArgumentParser(description="NFL Pagerank team rankings")
parser.add_argument('--in', action='store', dest='infile')
parser.add_argument('--out', action='store', dest='outfile')
parser.add_argument('--d', action='store', dest='d', type=float)
options = parser.parse_args()

N = 32.
ADJ = np.mat(np.zeros((N,N)))
PR = np.mat(np.zeros((N,1))) + 1./N

teams = {}
teamsA = []
wins = np.zeros(N)
losses = np.zeros(N)
fh = open(options.infile, 'rb')
reader = csv.reader(fh)
count = 0
panel = pd.Panel([], items=range(1,17))
for row in reader:
    if not teams.has_key(row[0]):
        teamsA.append(row[0])
        teams[row[0]] = len(teamsA) - 1
    if not teams.has_key(row[1]):
        teamsA.append(row[1])
        teams[row[1]] = len(teamsA) - 1
    wins[teams[row[0]]] += 1
    losses[teams[row[1]]] += 1
    ADJ[teams[row[0]], teams[row[1]]] += 1

rec_losses = 1 / losses
print ADJ
L = np.multiply(ADJ, rec_losses).T
print "L:\n", L
d = options.d
D = np.mat(np.zeros((N,1))) + (1. - d)/N
PR_last = PR * np.inf
print PR_last
for i in range(0,40):
    print "sum:  ", PR.sum()
    print "norm: ", np.linalg.norm(PR)
    PR_last = PR
    PR = D + d * ADJ * PR
    PR = PR / PR.sum()
df = pd.DataFrame( np.array(PR).T[0], index=teamsA, columns=['PageRank'])
df['W'] = wins
df['L'] = losses
df[['W', 'L']] = df[['W', 'L']].astype(int)
df['PageRank'] *= 100
dfs = df.sort(columns='PageRank', ascending=False)
print dfs
dfs.to_csv(options.outfile, sep='|',
           float_format='%2.2f', index_label='Team')
# PR = np.array(PR).T[0]
# zipped = zip(teamsA, PR)
# sortd = sorted(zipped, key=lambda x: x[1], reverse=True )
# print pandas.DataFrame(sortd)
# print "sorted: \n", sortd
# flattened = [item for sublist in PR for item in sublist]
# print "flattened: ", flattened
