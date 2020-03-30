import numpy as np
import matplotlib.pyplot as pplt
from matplotlib import gridspec, cm, colors
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
        
    def SN_hist(self, data, data_SN, upper_SN, lower_SN):
        low_SN = data[np.where(data_SN <= lower_SN)]
        high_SN = data[np.where(data_SN >= upper_SN)]
        low_SN_hist = np.histogram(low_SN)
        high_SN_hist = np.histogram(high_SN)
        binRange = np.concatenate((low_SN_hist[1],high_SN_hist[1]))
        binRange = np.linspace(np.min(binRange),np.max(binRange),100)
        return low_SN_hist, high_SN_hist, binRange
        
    def gauss(self, x, a, x0, var):
        return a*np.exp(-(x-x0)**2/(2*var))
    
    def normal(self, x, a, x0, var):
        return 1./(np.sqrt(2*var*np.pi))*np.exp(-(x-x0)**2/(2*var))
        
    def fitGauss(self, x, y, p0=[1,0,1]):
        popt, pcov = curve_fit(self.gauss,x,y,p0)
        return popt
    
    def stats(self,y):
        median = np.median(y)
        std_dev = np.sqrt(np.sum((y**2))/(len(y)-1))
        return median, std_dev

    def plotResults(self):
        #set fontsize
        pplt.rcParams.update({'font.size': 8})
        #set up figure axes
        widths = [4,1]
        gs_kw = dict(width_ratios=widths)
        fig, axs = pplt.subplots(3, 2, gridspec_kw=gs_kw)
        #plot params
        s = 1
        cmap = cm.get_cmap('bone_r')#'inferno_r'
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.2, b=1.0),cmap(np.linspace(0.2, 1.0, 100)))
        
        rgbahigh = new_cmap(0.9)
        rgbalow = new_cmap(0.1)
        ylabels = [r'$\Delta $Teff',r'$\Delta $log(g)',r'$\Delta $Fe']
        xlabels = ['Teff','log(g)','Fe']
        p0s = [(1,0,50**2),(1,0,0.1**2),(1,0,0.05**2)]
        xlims_real = [(3750,5750),(0,4.5),(-2.5,0.75)]
        xlims_synth = [(3000,6500),(-0.5,5.5),(-2.75,0.75)] #old vals (3500,8500),(-0.5,5.5),(-2.75,0.75)
        #set right xlim
        if np.max(self.targets[0]) <= 5500:
            xlims = xlims_real
        else:
            xlims = xlims_synth
        ylims = [1000,2,1]
        gausslims = [200,0.4,0.2]
        #plot
        for i in range(0,3):
            target = self.targets[i]
            resid = self.resids[i]
            #mean and scatter
            median, scatter = self.stats(resid)
            #gen and fit histograms
            hist = self.SN_hist(resid, self.target_SN, 200, 100)
            ##plot results##
            #plot resid
            plot = axs[i, 0].scatter(target, resid, c=self.target_SN, s=s, cmap=new_cmap, vmin=0, vmax=250,alpha=0.4)
            axs[i, 0].plot(np.linspace(xlims[i][0],xlims[i][1],10), np.zeros(10),'k',linewidth=0.5)
            axs[i, 0].set_ylim(-ylims[i],ylims[i])
            axs[i, 0].set_xlabel(xlabels[i])
            axs[i, 0].set_ylabel(ylabels[i])
            axs[i, 0].set_xlim(xlims[i][0],xlims[i][1])
            axs[i, 0].annotate(s=r'$\~m={:.3f}$   $s={:.3f}$'.format(median,scatter),xy=(xlims[i][0],-(ylims[i]*0.9)))
            try:
                gauss_out_low = self.fitGauss(hist[0][1][:-1], hist[0][0]/np.max(hist[0][0]), p0s[i])
                gauss_out_high = self.fitGauss(hist[1][1][:-1], hist[1][0]/np.max(hist[1][0]), p0s[i])
                gauss_ran = np.linspace(-gausslims[i],gausslims[i],100)
                gauss_low = self.normal(gauss_ran, gauss_out_low[0], gauss_out_low[1], gauss_out_low[2])
                gauss_high = self.normal(gauss_ran, gauss_out_high[0], gauss_out_high[1], gauss_out_high[2])
                #plot gauss
                axs[i, 1].plot(gauss_high, gauss_ran,color=rgbahigh,alpha=0.7)
                axs[i, 1].plot(gauss_low, gauss_ran,color=rgbalow,alpha=0.7)
                axs[i, 1].set_ylim(-gausslims[i],gausslims[i])
            except Exception as e:
                print(e)
                #plot histogram
                axs[i, 1].plot(hist[0][0],hist[0][1][:-1],color=rgbalow)
                axs[i, 1].plot(hist[1][0],hist[1][1][:-1],color=rgbahigh)
            axs[i, 1].xaxis.set_visible(False)
            axs[i, 1].yaxis.tick_right()
        
        pplt.tight_layout()
        #colorbar
        cbar = fig.colorbar(plot, ax=axs.ravel().tolist(), label='S/N', extend='both', pad=0.1)
        cbar.set_alpha(1)
        cbar.draw_all()
        #savefig
        pplt.savefig(self.respath + '/' + self.trialname + '.png', dpi=220)
        
    
