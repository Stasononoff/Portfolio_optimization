import networkx as nx
import math

from itertools import combinations
from operator import itemgetter
from timeit import default_timer
from random import shuffle

EPS = 1e-6

def exact_MWIS(graph, pi, b_score=0):
    ''' compute mawimum weighted independent set (recursively) using python
    networkx package. Input items are:
    - graph, a networkx graph
    - pi, a dictionary of dual values attached to node (primal constraints)
    - b_score, a bestscore (if non 0, it pruned some final branches)
    It returns:
    - mwis_set, a MWIS as a sorted tuple of nodes
    - mwis_weight, the sum over n in mwis_set of pi[n]'''
    global best_score
#     assert sum(pi.values()) > 0
    graph_copy = graph.copy()
    # mwis weight is stored as a 'score' graph attribute
    graph_copy.graph['score'] = 0
    best_score = b_score

    def get_mwis(G):
        '''compute mawimum weighted independent set (recursively) for non
        yet computed sets. Input is a networkx graph, output is the 
        exact MWIS set of nodes and its weight.
        Based on "A column generation approach for graph coloring" from
        Mehrotra and Trick, 1995, using recursion formula:
        MWIS(G union {i}) = max(MWIS(G), MWIS({i} union AN(i)) where
        AN(i) is the anti-neighbor set of node i'''
        global best_score
        # score stores the best score along the path explored so far
        key = tuple(sorted(G.nodes()))
        ub = sum(pi[n] for n in key)
        score = G.graph['score']
        # if graph is composed of singletons, leave now
        if G.number_of_edges == 0:
            if score + ub > best_score + EPS:
                best_score = score + ub
            return key, ub
        # compute highest priority node (used in recursion to choose {i})
        node_iter = ((n, deg*pi[n]) for (n, deg) in G.degree())
        node_chosen, _ = max(node_iter)
        pi_chosen = pi[node_chosen]
        node_chosen_neighbors = list(G[node_chosen])
        pi_neighbors = sum(pi[n] for n in node_chosen_neighbors)
        G.remove_node(node_chosen)
        # Gh = G - {node_chosen} union {anti-neighbors{node-chosen}}
        # For Gh, ub decreases by value of pi over neighbors of {node_chosen}
        # and value of pi over {node_chosen} as node_chosen is disconnected
        # For Gh, score increases by value of pi over {node_chosen}
        Gh = G.copy()
        Gh.remove_nodes_from(node_chosen_neighbors)
        mwis_set_h, mwis_weight_h = tuple(), 0
        if Gh:
            ubh = ub - pi_neighbors - pi_chosen
            if score + pi_chosen + ubh > best_score + EPS:
                Gh.graph['score'] += pi_chosen
                mwis_set_h, mwis_weight_h = get_mwis(Gh)
            del Gh
        mwis_set_h += (node_chosen, )
        mwis_weight_h += pi_chosen
        # Gp = G - {node_chosen}
        # For Gp, ub decreases by value of pi over {node_chosen}
        # For Gh, score does not increase
        mwis_set_p, mwis_weight_p = tuple(), 0
        if G:
            ubp = ub - pi_chosen
            if score + ubp > best_score + EPS:
                mwis_set_p, mwis_weight_p = get_mwis(G)
            del G
        # select case with maximum score
        if mwis_set_p and mwis_weight_p > mwis_weight_h + EPS:
            mwis_set, mwis_weight = mwis_set_p, mwis_weight_p
        else:
            mwis_set, mwis_weight = mwis_set_h, mwis_weight_h
        # increase score
        score += mwis_weight
        if score > best_score + EPS:
            best_score = score
        # return set and weight
        key = tuple(sorted(mwis_set))
        return key, mwis_weight

    return get_mwis(graph_copy)