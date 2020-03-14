import numpy as np
import matplotlib.pyplot as pplt
from matplotlib import gridspec, cm
from scipy.optimize import curve_fit

#for now gen fake data
x = np.arange(0,1000)
y = np.ones(1000)
y *= np.random.randn(1000) #normal dist about 0
fake_SN = np.random.random(1000) #random nums between 0 and 1
y *= fake_SN

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fitGauss(gauss, x, y, p0=[1,0,1]):
    popt, pcov = curve_fit(gauss,x,y,p0)
    return popt

def plotResults(x,y,fake_SN): ###NOT READY-NEED DATA FORMAT
    ###Gen Histrograms###
    low_SN = y[np.where(np.abs(fake_SN) >= 0.6)]
    low_SN_hist = np.histogram(low_SN,bins=50)
    high_SN = y[np.where(np.abs(fake_SN) <= 0.4)]
    high_SN_hist = np.histogram(high_SN,bins=50) #out (hist values, bins)
    binRange = np.concatenate((low_SN_hist[1],high_SN_hist[1]))
    
    ###Fit Gaussians###
    lowa, lowx0, lowsig = fitGauss(gauss,low_SN_hist[1][:-1],low_SN_hist[0])
    higha, highx0, highsig = fitGauss(gauss,high_SN_hist[1][:-1],high_SN_hist[0])
    xran = np.linspace(np.min(binRange),np.max(binRange),100)
    lowGauss = gauss(xran,lowa,lowx0,lowsig)
    highGauss = gauss(xran,higha,highx0,highsig)
    
    ###PLOT###
    pplt.rcParams.update({'font.size': 8}) #set fontsize
    
    #set up figure axes
    widths = [4,1]
    gs_kw = dict(width_ratios=widths)
    fig, axs = pplt.subplots(3, 2, gridspec_kw=gs_kw)
    
    #plot params
    s = 3
    cmap = 'magma'
    cmap = cm.get_cmap('magma')
    rgba1 = cmap(0.9)
    rgba2 = cmap(0.1)
    
    #plot results
    plot = axs[0, 0].scatter(x, y,c=fake_SN, s=s, cmap=cmap)
    axs[0, 0].set_xlabel('xlabel 1')
    axs[0, 0].set_ylabel('ylabel 1')

    #axs[0, 1].plot(low_SN_hist[0], low_SN_hist[1][:-1],color=rgba1)
    #axs[0, 1].plot(high_SN_hist[0], high_SN_hist[1][:-1],color=rgba2)
    axs[0, 1].plot(lowGauss, xran,color=rgba1)
    axs[0, 1].plot(highGauss, xran,color=rgba2)
    axs[0, 1].xaxis.set_visible(False)
    axs[0, 1].yaxis.tick_right()

    axs[1, 0].scatter(x, y,c=fake_SN, s=s, cmap=cmap)
    axs[1, 0].set_xlabel('xlabel 2')
    axs[1, 0].set_ylabel('ylabel 2')

    #axs[1, 1].plot(low_SN_hist[0], low_SN_hist[1][:-1],color=rgba1)
    #axs[1, 1].plot(high_SN_hist[0], high_SN_hist[1][:-1],color=rgba2)
    axs[1, 1].plot(lowGauss, xran,color=rgba1)
    axs[1, 1].plot(highGauss, xran,color=rgba2)
    axs[1, 1].xaxis.set_visible(False)
    axs[1, 1].yaxis.tick_right()

    axs[2, 0].scatter(x, y,c=fake_SN, s=s, cmap=cmap)
    axs[2, 0].set_xlabel('xlabel 3')
    axs[2, 0].set_ylabel('ylabel 3')

    #axs[2, 1].plot(low_SN_hist[0], low_SN_hist[1][:-1],color=rgba1)
    #axs[2, 1].plot(high_SN_hist[0], high_SN_hist[1][:-1],color=rgba2)
    axs[2, 1].plot(lowGauss, xran,color=rgba1)
    axs[2, 1].plot(highGauss, xran,color=rgba2)
    axs[2, 1].xaxis.set_visible(False)
    axs[2, 1].yaxis.tick_right()

    pplt.tight_layout()

    fig.colorbar(plot, ax=axs.ravel().tolist(),label='N/S')
        
    pplt.show()
    
plotResults(x,y,fake_SN)
