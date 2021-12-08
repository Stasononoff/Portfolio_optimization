import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle

import neal
import dimod

import networkx as nx
import math

from itertools import combinations
from operator import itemgetter
from timeit import default_timer
from random import shuffle

EPS = 1e-6
R0 = 0.0192

r0 = np.log(np.exp(R0)**(1/365))
r0


class Market(object):
    
    def __init__(self, tickers, start_date= '2010-01-01', end_date = '2019-01-01'):
        
        print(start_date, end_date)
        self.data = yf.download(tickers, start = start_date, end = end_date).Close.fillna(method = 'bfill')#.dropna()#
        tickers = np.array(tickers)
        print(self.data.shape)
        
        # отбираем только фонды с положительным доходом
        change_data = self.data/self.data.shift(periods=1)
        market_profit = np.log(change_data).mean()
        self.change_data = np.log(change_data.T[market_profit > EPS].T)
#         self.change_data = np.log(change_data)
#         print(len(self.change_data.T), len(change_data.T), market_profit)
        
        # Рассчитываем доходность и риск
        self.N = len(self.change_data.T)
        self.market_profit = self.change_data.mean()
        self.market_risk = self.change_data.cov()
        self.tickers = list(self.change_data.columns) #list(tickers[market_profit > (1+EPS)])
        
        self.v_list = (self.market_profit) - 0.2*np.diagonal(self.market_risk)
        self.e_matrix = self.change_data.corr()
        
        
        self.G = None
        self.vert_weights = None
        self.wmis_mask = None
        self.wmis_G = None
        self.J = None
        self.h = None
        self.g = None
        
        self.B = 1
        
        print('Market size: {}'.format(self.N))
        
    def build_Ising_hamiltonian(self, theta, weighted = False, market_num = 3 ):
        
        B = self.B

        # equal_weighted
        if weighted == False:
            A_list = np.array(([1] * self.N))

        # Sharp_weighted
        elif weighted == True:
            A_list = (self.market_profit/np.diagonal(self.market_risk)).values

        # Max share of investments
        A_list = A_list/market_num
        A_matrix = np.kron(np.reshape(A_list, [len(A_list),1]), A_list)

        Cov = self.market_risk.values
        P = self.market_profit.values
        
        # Hamiltonian parts
        J_matrix = 1/4*(theta[1]*Cov + theta[2]*A_matrix)
        h_list = 1/2*(theta[1]*np.sum(Cov, axis = 0) + theta[2]*np.sum(A_matrix, axis = 0) - theta[0]*P - 2*B*theta[2]*A_list)  
        g =  - theta[0]/2*np.sum(P) + theta[1]/4*np.sum(Cov) + theta[2]/4*(np.sum(A_matrix) - 4*B*np.sum(A_list) + 4*B**2)


        J = {}

        for i in range(len(J_matrix)):
            for j in range(len(J_matrix)):
#                 if j>i:
                J.update({(i,j): J_matrix[i][j]})

        h = dict(zip(list(range(len(h_list))), list(h_list)))
        
        self.J = J
        self.h = h
        self.g = g
        
    def build_market_graph(self, edge = 0.25):
        
        edge = np.log(np.exp(edge)**(1/365))
        
        ind_vert_list = []
        ind_vert_weight = []
        
        clist = []
        elist = []
        v_list = []
        for c1, p  in zip(self.change_data.cov().to_dict().keys(), self.v_list):
            clist.append(c1)

            i = 0
            for c2, corr in self.change_data.cov().to_dict()[c1].items():
                if ((corr > edge) & (c1 != c2)):
                    i += 1
                if ((clist.count(c2)==0) & (corr > edge) & (c1 != c2)):
                    elist.append((c1, c2, corr))
                    
            if i == 0:
                ind_vert_list.append(c1)
                ind_vert_weight.append(p)
            else:
                v_list.append(p)
                

        G=nx.Graph()
        G.add_weighted_edges_from(elist)
        G.add_nodes_from(ind_vert_list)
        v_list.extend(ind_vert_weight)
        self.G = G
        self.vert_weights = dict(zip(G.nodes(), v_list))
        
        
    def draw_market(self):
        nx.draw_networkx(self.G)
        
    def get_market_profit(self):
        return self.market_profit
    
    def get_market_risk(self):
        return self.market_risk
    
    def get_data(self):
        return self.data
    
    def get_change_data(self):
        return self.change_data
    

    


class Portfolio(Market):
    
    def __init__(self, market):
        
        self.tickers = market.tickers
        self.data = market.data
        self.N = market.N
        
        self.change_data = market.change_data
        self.market_profit = market.market_profit
        self.market_risk = market.market_risk
        self.v_list = market.v_list
        self.e_matrix = market.e_matrix
        
        self.G = market.G
        self.vert_weights = market.vert_weights
        self.wmis_mask = market.wmis_mask
        self.wmis_G = market.wmis_G
        
        self.J = market.J
        self.h = market.h
        self.g = market.g
        
        self.B = market.B

        self.weights = None
        self.profit = None
        self.risk = None
        self.Sharp_koef = None
        self.mask = None
        self.rho = 1
        self.gamma = 0.1
        self.cost = 2000
        
#         self.mask = None
        
        
    def build_portfolio(self, weights = None):
        if weights == None:
#             weights = np.abs(np.random.randn(self.N))
            weights = np.abs(np.random.rand(self.N))
        else:
            weights = np.array(weights)
        self.weights = weights / np.sum(weights)
        
        weights[weights>0] = 1
        self.mask = weights
        
        self.profit = np.dot(self.market_profit,self.weights)
        self.risk = np.dot(np.dot(self.weights, self.market_risk.values), self.weights)
#         self.Sharp_koef = (np.log(np.exp(self.profit)**365) - R0)/(np.sqrt(self.risk*365))
        self.Sharp_koef = ((1+self.profit)**365 - 1 - R0)/(np.sqrt(self.risk*365))
        self.cost = -self.profit + self.gamma*self.risk
        
        
#         print(self.weights, self.risk, self.profit, self.Sharp_koef)
        
    def get_weights(self):
        return self.weights
    
    def get_profit(self, period = 1):
        if period == 1:
            return self.profit
#         return np.log(np.exp(self.profit)**period)
        return (1+self.profit)**period - 1
    
    def get_risk(self, period = 1):
        return np.sqrt(self.risk*period)
#         return self.risk*np.sqrt(period)
    
    def get_Sharp(self):
        return self.Sharp_koef
    
    def get_cost(self):
        return self.cost
    
    def get_mask(self):
        return self.mask
    
    def get_hamiltonian(self, J = True, h = True, g = False):
        return self.J, self.h
    
    def build_binary_portfolio(self, weighted = False, alpha = 0 , mask = None):
        
        
        if mask == None:
            mask = np.random.randn(self.N)
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
        else:
            mask = np.array(mask)
        
        if weighted == True:
            weights = mask * (self.market_profit*np.diag(self.market_risk.values) + alpha)
        else:
            weights = mask
            
        self.weights = weights / np.sum(weights)
        weights[weights>0] = 1
        self.mask = weights
        
        
        
        self.profit = np.dot(self.market_profit,self.weights)
        self.risk = np.dot(np.dot(self.weights, self.market_risk.values),self.weights)
#         self.Sharp_koef = (np.log(np.exp(self.profit)**365) - R0)/(np.sqrt(self.risk*365))
        self.Sharp_koef = ((1+self.profit)**365 - 1 - R0)/(np.sqrt(self.risk*365))
        self.cost = -self.profit + self.gamma*self.risk
        
#         print(mask, self.weights, self.risk, self.profit, self.Sharp_koef)

    def get_portfolio(self):
        res = pd.DataFrame(np.array([self.tickers,self.weights]).T, columns = ['ticker', 'weight'])
        res['weight'] = res['weight'].astype('float32')
        return res[res['weight']>0]
        
        
    

        
        
    
    
def get_time(function, params = None):
    start_time = time.time()
    
    if params != None:
        data_out = function(**params)
    else:
        data_out = function()
        
    dt = (time.time() - start_time)
    print("--- %s seconds ---" % (dt))
    return data_out, dt
    

        