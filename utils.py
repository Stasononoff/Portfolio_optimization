from market import *

with open('ETF_list.pkl', 'rb') as fp:
    ETF_list = pickle.load(fp)
    
with open('crypto_list.pkl', 'rb') as fp:
    crypto_list = pickle.load(fp)
    
def get_time(function, params = None):
    start_time = time.time()
    
    if params != None:
        data_out = function(**params)
    else:
        data_out = function()
        
    dt = (time.time() - start_time)
    print("--- %s seconds ---" % (dt))
    return data_out, dt

def build_market(sample_num = 16, type_ = 'full'):
    
    if type_ == 'full':

        ETF_num = int(sample_num*len(ETF_list)/(len(ETF_list) + len(crypto_list)))
        crypto_num = int(sample_num*len(crypto_list)/(len(ETF_list) + len(crypto_list)))
        sample_ticker = ETF_list[:ETF_num] + crypto_list[:crypto_num]
        print(sample_ticker, ETF_num, crypto_num)
        
    elif type_ == 'ETF':
        
        ETF_num = sample_num
        sample_ticker = ETF_list[:ETF_num]
        print(sample_ticker, ETF_num)
    
    
    market, t = get_time(Market, params = {'tickers' : sample_ticker, 'start_date' : '2018-08-24', 'end_date' : '2021-08-24'})
    portfolio, t = get_time(Portfolio, {'market' : market})
    portfolio.build_binary_portfolio(weighted = False, alpha = 0)
    return market, portfolio

def drow_cloud(portfolio, N = 4000):

    Sharp = np.zeros(N)
    risk = np.zeros(N)
    doh = np.zeros(N)
    portf = np.zeros((N, portfolio.N))

    for n in range(N):
        portfolio.build_portfolio()
#         portfolio.build_binary_portfolio()


        portf[n,:] = portfolio.get_weights()
        risk[n] = portfolio.get_risk(period = 365)
        doh[n] = portfolio.get_profit(period = 365)
        Sharp[n] = portfolio.get_Sharp()

    old_risk = risk.copy()

#     risk = risk[(old_risk<1)&(doh<1)]
#     doh = doh[(old_risk<1)&(doh<1)]

    del old_risk

    plt.scatter(risk*100,doh*100,c=list((doh-R0)/risk),marker='.')
    plt.xlabel('риск, %')
    plt.grid()
    plt.ylabel('доходность, %')
    
    
def plot_beautyful(x_arr, y_arr, x_label = None, y_label = None, title = None, fontsize = 15,
                   title_fontsize = 17, figsize = (6,6), labelsize = 16,
                   markersize = 8, marker = 'o', linewidth = 3, label = None, show = False, set_fig = False):
    if set_fig == True:
        plt.figure(figsize=figsize, dpi=160)
    plt.plot(x_arr, y_arr, linewidth = linewidth, marker = marker, markersize = markersize, label = label)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(fontsize = labelsize)
    plt.yticks(fontsize = labelsize)
    plt.title(title, fontsize=title_fontsize)
    plt.grid()
    if show == True:   
        plt.show()
    
    
    
def exact_solution(portfolio, weighted = False, alpha = 0, fixed_param = 'risk', gamma_val = 0.1, rho_val = 1):
    portfolio.gamma = gamma_val
    portfolio.rho = rho_val
    best_mask = portfolio.mask
    

    if fixed_param == None:
        p = 0
        for i in range(1, 2**portfolio.N):
            m = list(map(int, list(bin(i)[2:])))
            mask = [0]*(portfolio.N - len(m)) + m
            portfolio.build_binary_portfolio(weighted = weighted, alpha = alpha, mask = mask)
            if portfolio.get_Sharp() > p:
                p  = portfolio.get_Sharp()
                weights = portfolio.get_weights()
                best_mask = mask
        
    elif fixed_param == 'risk':
        p = 100
        for i in range(1, 2**portfolio.N):
            m = list(map(int, list(bin(i)[2:])))
            mask = [0]*(portfolio.N - len(m)) + m
            portfolio.build_binary_portfolio(weighted = weighted, alpha = alpha, mask = mask)
            if portfolio.get_cost() < p:
                p  = portfolio.get_cost()
                weights = portfolio.get_weights()
                best_mask = mask
    portfolio.build_binary_portfolio(weighted = weighted, alpha = alpha, mask = best_mask)
        
        
        
def get_effective_boundary(portfolio, weighted = False, get_time = True, name = None, size = None):

    risk_list = []
    profit_list = []
    sharp_list = []
    gamma_list = []
    time_list = []
    name_list = []
    size_list = []
    mask_list = []
    ticker_list = []
    
    f = 40**(1/40)

    for i in range(1,41):
        t0 = time.time()
        exact_solution(portfolio, weighted = weighted, alpha = 0, gamma_val = f**i, rho_val = 30)
        dt = time.time() - t0
        mask_list.append(portfolio.get_mask())
        risk_list.append(portfolio.get_risk(period = 365)*100)
        profit_list.append(portfolio.get_profit(period = 365)*100)
        gamma_list.append(i)
        sharp_list.append(float(portfolio.get_Sharp()))
        time_list.append(dt)
        size_list.append(size)
        name_list.append(name)
        ticker_list.append(portfolio.tickers)
        
    r = np.array([gamma_list, risk_list, profit_list, sharp_list, size_list, time_list])
    df = pd.DataFrame(r.T, columns = ['gamma', 'risk','return','sharp', 'size' ,'time'])
    df['name'] = name
    df['mask'] = mask_list
    df['tickers'] = ticker_list
        
    return  df

def build_ising_effective_boundary(portfolio, sampler, weighted = False, market_num = 10, theta = [1,1,1e-3], get_time = True, name = None, size = None):

    risk_list = []
    profit_list = []
    sharp_list = []
    gamma_list = []
    mask_list = []
    time_list = []
    name_list = []
    size_list = []
    ticker_list = []
    
    g_start = theta[0]
    g_end = theta[1]
    a = (g_end/g_start)**(1/100)
    
    theta[0] = 1
    
    for i in range(1,101):
        
        
        
        theta[1] = theta[1]/a
        
#         f = theta[0]**(1/40)

#         theta[0] = 1

#         theta[1] = f**(i - 41)

#         theta[2] = 1e-3


        portfolio.build_Ising_hamiltonian(theta, weighted = False, market_num = market_num)
        
        
        J,h = portfolio.get_hamiltonian()
#         print(h,J)

        t0 = time.time()
#         print(name)
        sampleset = sampler.sample_ising(h, J)
        dt = time.time() - t0
        
        mask = np.array(list(sampleset.first.sample.values()))
        mask[mask == -1] = 0

        portfolio.build_binary_portfolio(weighted = False, alpha = 0 , mask = list(mask))
#             weight_list.append(portfolio.get_weights())

        mask_list.append(mask)
        risk_list.append(portfolio.get_risk(period = 365)*100)
        profit_list.append(portfolio.get_profit(period = 365)*100)
        gamma_list.append(theta[1])
        sharp_list.append(portfolio.get_Sharp())
        time_list.append(dt)
        size_list.append(size)
        name_list.append(name)
        ticker_list.append(portfolio.tickers)
            
    r = np.array([gamma_list, risk_list, profit_list, sharp_list, size_list, time_list])
    df = pd.DataFrame(r.T, columns = ['gamma', 'risk','return','sharp', 'size' ,'time'])
    df['name'] = name
    df['mask'] = mask_list
    df['tickers'] = ticker_list
    
            
           

    return df