from  nflteams import NFLTeams
import numpy as np
import csv
import argparse as ap
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm
from tabulate import tabulate
from operator import itemgetter
from pprint import pprint

def create_win_graph(csv_file):
    g = nx.DiGraph()
    nt = NFLTeams()
    for row in csv.DictReader(open(csv_file, 'rb')):
        winner = nt.get_team(row['Winner/tie'])
        loser = nt.get_team(row['Loser/tie'])
        g.has_node(winner) or g.add_node(winner)
        g.has_node(loser) or g.add_node(loser)
        diff = int(row['PtsW']) - int(row['PtsL'])
        g.add_edge(loser, winner, W=row['W#'], PtsW=row['PtsW'],
                PtsL=row['PtsL'], YdsW=row['YdsW'], TOW=row['TOW'],
                YdsL=row['YdsL'],TOL=row['TOL'], diff=diff)
    return nt, g

def show_sorted_centrality(centrality):
    sortd = sorted(centrality.items(), key=itemgetter(1), reverse=True)
    print tabulate([[i[0].name,i[1]] for i in sortd], floatfmt='.3f')

def aligned_array(*args):
    arr = []
    keys = args[0].keys()
    for key in keys:
        row = [key]
        for dic in args:
            if dic.has_key(key):
                row.append(dic[key])
            else:
                row.append('apple')
        arr.append(row)
    return arr

def convert_to_name(table):
    return [[row[0].name] + row[1:] for row in table]

def katz(g):
    eigv = eig(nx.adjacency_matrix(g, weight='diff').todense())
    max_eigv = max(eigv[0])
    max_eigv_reciprocal = 1./max_eigv
    alpha = max_eigv_reciprocal
    alpha = 0.9 * alpha
    beta = 1 - alpha
    katz_centrality = nx.katz_centrality(g, alpha=alpha, beta=beta,
                                         weight='diff')
    return katz_centrality

# def pageRank(g, d=0.85, weight=None):
    # node_list = g.nodes()
    # ADJ_options = {}
    # if weight:
        # ADJ_options["weight"] = weight
    # ADJ = nx.adjacency_matrix(g,**ADJ_options).todense().T
    # N = len(node_list)
    # PR = np.zeros((N,1)) + 1./N
    # losses = g.out_degree()
    # losses = np.array([losses[node] for node in node_list])
    # rec_losses = 1 / losses
    # L = np.multiply(ADJ, rec_losses).T
    # D = np.zeros((N,1)) + (1. - d)/N
    # for i in range(0,40):
        # PR = D + d * ADJ * PR
        # PR = PR / PR.sum()
    # PR = np.array(PR)
    # return dict(zip(node_list, PR.T[0]))

def pageRank(A, order, max_iter=1000, epsilon=1e-5, d=0.85):
    N = len(A)
    order = [v if v != 0 else 1 for v in order]
    order = np.array(order)
    rec = 1. / order
    L = np.multiply(A.T, rec).T
    PR = np.zeros((N,1)) + 1./N
    D = np.zeros((N,1)) + (1. - d)/N
    old_PR = PR
    for i in range(0,max_iter):
        old_PR = PR
        PR = D + d * L * PR
        if norm(old_PR - PR) < epsilon:
            break
    PR = PR / PR.sum()
    return PR.ravel().tolist()[0]

def WLRank(nt, g, d=0.85, weight=None, epsilon=0.00001, max_iter=1000):
    node_list = g.nodes()
    ADJ_options = {}
    if weight:
        ADJ_options["weight"] = weight
    # rows are losses, columns are wins
    LA = nx.adjacency_matrix(g,**ADJ_options).todense()
    WA = LA.T #rows are wins, columns are losses
    nt.add_attrs('wins', g.out_degree())
    wins = nt.get_attrs('wins', ordered_by=node_list)
    PRl = pageRank(LA, wins, max_iter=max_iter, epsilon=epsilon, d=d)

    nt.add_attrs('PRl', PRl, ordered_by=node_list)

    nt.add_attrs('losses', g.out_degree())
    losses = nt.get_attrs('losses', ordered_by=node_list)
    PRw = pageRank(WA, losses, max_iter=max_iter, epsilon=epsilon, d=d)
    nt.add_attrs('PRw', PRw, ordered_by=node_list)

    return nt
def visualize(g, node_scale=None, **kwargs):
    if node_scale:
        max_v = max(node_scale.items(), key=lambda x: x[1])[1]
        coef = 500./max_v
        node_size = [coef * centrality[node] for node in g]
    labels = {k: k.PFR() for k in g}
    plt.figure(figsize=(20,20))
    nx.draw_networkx(g, nx.spring_layout(g,k=2/np.sqrt(32)), labels=labels,
                     node_size=node_size, node_color='#DAF0F5',
                     edge_color='#D7DFE0')
    plt.show()

def normalize(d):
    values = np.array(d.values())
    sum_v = values.sum()
    return {k: 100 * v / sum_v for k,v in d.items()}


############
#  script  #
############

if __name__ == '__main__':

    parser = ap.ArgumentParser(description="NFL Pagerank team rankings")
    parser.add_argument('--in', action='store', dest='infile')
    parser.add_argument('--out', action='store', dest='outfile')
    parser.add_argument('--d', action='store', dest='d', type=float)
    options = parser.parse_args()

    nt, g = create_win_graph(options.infile)
    nt = WLRank(nt, g)
    prw = np.array(nt.get_attrs('PRw'))
    prl = np.array(nt.get_attrs('PRl'))
    PR = (prw - prl) * 100
    print "\nPR:\n", PR
    nt.add_attrs('PR', PR)
    nt.set_attrs('PRw', prw * 100)
    nt.set_attrs('PRl', prl * 100)

    for team in nt.teams:
        beat_ = ''
        for edge in g.in_edges(team):
            beat_team = edge[0]
            beat_ += ' ' + beat_team.flair + '(' +\
                     '{: .2f}'.format(beat_team.PR) + ')'
        beaten_ = ''
        for edge in g.out_edges(team):
            beaten_team = edge[1]
            beaten_ += ' ' + beaten_team.flair + '(' +\
                        '{: .2f}'.format(beaten_team.PR) + ')'
        team.beat   = beat_
        team.beaten = beaten_

    table = nt.get_table(columns=['flair', 'PR', 'PRw', 'PRl', 'beat',
                                        'beaten'], sorted_by=1)
    table.reverse()

    headers = ['Team', 'PR', 'PRw', 'PRl', 'Beat', 'Beaten by']

    print tabulate(table, floatfmt='.2f', tablefmt='pipe', headers=headers)
    import sys; sys.exit()

    rows = []
    for key in pr.keys():
        row = [key.flair]
        row.append(100 * pr[key])
        for edge in g.in_edges(key):
            neighbor = edge[0]
            row.append(neighbor.flair)
            row.append(100. * pr[neighbor])
        rows.append(row)

    max_ = len(max(rows, key=lambda x: len(x)))
    rows = [row + [None]*(max_ - len(row)) for row in rows]
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    headers = ['Team', 'PR'] + ['beat', 'PR']*((max_ - 2)/2)
    print tabulate(rows, headers=headers, floatfmt='.2f', tablefmt='pipe')

    g1 = g.copy()
    edges = g.edges()
    Ws = [int(g.get_edge_data(*e)['W']) for e in g.edges()]
    max_W = max(Ws)
    gs = {max_W: g}
    for w in reversed(range(1,max_W)):
        gs[w] = gs[w + 1].copy()
        for edge in gs[w].edges():
            if int(gs[w].get_edge_data(*edge)['W']) > w:
                gs[w].remove_edge(*edge)

        Wst = [int(gs[w].get_edge_data(*e)['W']) for e in gs[w].edges()]
        max_Wt = max(Wst)
    gs = [gs[k] for k in sorted(gs.keys())]
    # for g in gs:
        # print "\nlen(g.edges()):\n", len(g.edges())
    wins = g.in_degree()
    losses = g.out_degree()

    #katz centrality
    # g_katz = katz(g)
    # katz_arr = aligned_array(katz_arr,wins, losses)
    # katz_arr = convert_to_name(katz_arr)

    # print tabulate(katz_arr, floatfmt='.2f')

    # eigenvalue centrality
    # eigen = nx.eigenvector_centrality(g, max_iter=1000)
    # show_sorted_centrality(centrality)

    # visualize(g, node_scale=centrality)
    aligning = []
    for g in reversed(gs):
        p = {k.name: v for k,v in pageRank(g).items()}
        p = normalize(p)
        aligning.append(p)
    aligning.append({k.name: v for k,v in wins.items()})
    aligning.append({k.name: v for k,v in losses.items()})
    # aligning.extend([wins, losses])
    out = aligned_array(*aligning)

    # pr = pageRank(g)
    # pr = normalize(pr)
    # g_katz = normalize(g_katz)
    # eigen = normalize(eigen)
    # out = aligned_array(pr, g_katz, eigen, wins,losses)

    # out = convert_to_name(out)
    out = sorted(out, key=lambda x: x[1], reverse=True)
    weeks = range(1, len(gs)+1)
    weeks.reverse()
    headers = ["Team"] + weeks + ["Wins", "Losses"]
    print tabulate(out, headers=headers, floatfmt='.2f', tablefmt="pipe")
    # dfs = df.sort(columns='PageRank', ascending=False)
    # print dfs
    # dfs.to_csv(options.outfile, sep='|',
            # float_format='%2.2f', index_label='Team')
# PR = np.array(PR).T[0]
# zipped = zip(teamsA, PR)
# sortd = sorted(zipped, key=lambda x: x[1], reverse=True )
# print pandas.DataFrame(sortd)
# print "sorted: \n", sortd
# flattened = [item for sublist in PR for item in sublist]
# print "flattened: ", flattened

