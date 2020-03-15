import numpy as np
import matplotlib.pyplot as pplt
from matplotlib import gridspec, cm
from scipy.optimize import curve_fit

class post_processing():
    def __init__(self, model_results, model_targets, respath, trialname):
        
        model_Fe = model_results[:,0]
        model_g = model_results[:,1]
        model_Teff = model_results[:,2]
        
        target_Fe = model_targets[:,0]
        target_g = model_targets[:,1]
        target_Teff = model_targets[:,2]
        
        Fe_resid = model_Fe - target_Fe
        g_resid = model_g - target_g
        Teff_resid = model_Teff - target_Teff
        
        self.resids = [Teff_resid, g_resid, Fe_resid]
        self.targets = [target_Teff, target_g, target_Fe]
        self.target_SN = model_targets[:,3]
        self.trialname = trialname
        self.respath = respath
        
    def SN_hist(self, data, data_SN, upper_SN, lower_SN,bins=50):
        low_SN = data[np.where(data_SN <= lower_SN)]
        high_SN = data[np.where(data_SN >= upper_SN)]
        low_SN_hist = np.histogram(low_SN,bins)
        high_SN_hist = np.histogram(high_SN,bins)
        binRange = np.concatenate((low_SN_hist[1],high_SN_hist[1]))
        binRange = np.linspace(np.min(binRange),np.max(binRange),100)
        return low_SN_hist, high_SN_hist, binRange
        
    def gauss(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def fitGauss(self, x, y, p0=[1,0,1]):
        popt, pcov = curve_fit(self.gauss,x,y,p0)
        return popt

    def plotResults(self):
        #set fontsize
        pplt.rcParams.update({'font.size': 8})
        #set up figure axes
        widths = [4,1]
        gs_kw = dict(width_ratios=widths)
        fig, axs = pplt.subplots(3, 2, gridspec_kw=gs_kw)
        #plot params
        s = 1
        cmap = cm.get_cmap('magma_r')
        rgba1 = cmap(0.9)
        rgba2 = cmap(0.1)
        ylabels = ['Teff','log(g)','Fe']
        xlabels = ['Teff','log(g)','Fe']
        ylims = [1000,2,1]
        #plot
        for i in range(0,3):
            target = self.targets[i]
            resid = self.resids[i]
            #gen and fit histograms
            hist = self.SN_hist(resid, self.target_SN, 200, 100)
            gauss_out_low = self.fitGauss(hist[0][1][:-1], hist[0][0])
            gauss_out_high = self.fitGauss(hist[1][1][:-1], hist[1][0])
            gauss_ran = hist[2]
            gauss_low = self.gauss(gauss_ran, gauss_out_low[0], gauss_out_low[1], gauss_out_low[2])
            gauss_high = self.gauss(gauss_ran, gauss_out_high[0], gauss_out_high[1], gauss_out_high[2])
            ##plot results##
            #plot resid
            plot = axs[i, 0].scatter(target, resid, c=self.target_SN, s=s, cmap=cmap, vmin=50, vmax=250)
            axs[i, 0].set_ylim(-ylims[i],ylims[i])
            axs[i, 0].set_xlabel(xlabels[i])
            axs[i, 0].set_ylabel(ylabels[i])
            #plot gauss
            axs[i, 1].plot(gauss_low, gauss_ran,color=rgba1)
            axs[i, 1].plot(gauss_high, gauss_ran,color=rgba2)
            axs[i, 1].xaxis.set_visible(False)
            axs[i, 1].yaxis.tick_right()
        
        pplt.tight_layout()
        #colorbar
        cbar = fig.colorbar(plot, ax=axs.ravel().tolist(), label='S/N', extend='both', pad=0.1)
        #savefig
        pplt.savefig(self.respath + '/' + self.trialname + '.pdf')
        
    
