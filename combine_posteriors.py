import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys
from dpgmm import *
import multiprocessing as mp
from scipy.misc import logsumexp
from cosmology import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import cPickle as pickle
import readdata

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def FindHeightForLevel(inArr, adLevels):
    # flatten the array
    oldshape = np.shape(inArr)
    adInput= np.reshape(inArr,oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)
    
    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted
    
    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in xrange(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]
    
    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])
    
    adHeights = np.array(adHeights)

    return adHeights

def initialise_dpgmm(dims,posterior_samples):
    model = DPGMM(dims)
    for point in posterior_samples:
        model.add(point)

    model.setPrior()
    model.setThreshold(1e-4)
    model.setConcGamma(1.0,1.0)
    return model

def compute_dpgmm(model,max_sticks=16):
    solve_args = [(nc, model) for nc in xrange(1, max_sticks+1)]
    solve_results = pool.map(solve_dpgmm, solve_args)
    scores = np.array([r[1] for r in solve_results])
    model = (solve_results[scores.argmax()][-1])
    print "best model has ",scores.argmax()+1,"components"
    return model.intMixture()

def evaluate_grid(density,x,y):
    sys.stderr.write("computing log posterior for %d grid points\n"%(len(x)*len(y)))
    sample_args = ((density,xi,yi) for xi in x for yi in y)
    results = pool.map(sample_dpgmm, sample_args)
    return np.array([r for r in results]).reshape(len(x),len(y))

def sample_dpgmm(args):
    (dpgmm,x,y) = args
    logPs = [np.log(dpgmm[0][ind])+prob.logProb([x,y]) for ind,prob in enumerate(dpgmm[1])]
    return logsumexp(logPs)

def solve_dpgmm(args):
    (nc, model) = args
    for _ in xrange(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

def rescaled_om(om,min_om,max_om):
    return (om - min_om)/(max_om-min_om)

def logit(y):
    return np.log(y/(1.0-y))

def jacobian(om,min_om,max_om):
    y = rescaled_om(om,min_om,max_om)
    return np.abs(1.0/(max_om-min_om)*1.0/(y*(1.0-y)))

def logjacobian(om,min_om,max_om):
    y = rescaled_om(om,min_om,max_om)
    return -np.log(max_om-min_om)-np.log(y*(1.0-y))
#    return np.abs(1.0/()*1.0/(y*(1.0-y)))

def renormalise(logpdf,dx,dy):
    pdf = np.exp(logpdf)
    return pdf/(pdf*dx*dy).sum()

def marginalise(pdf,dx,axis):
    return np.sum(pdf*dx,axis=axis)

def reflective_boundaries(x,a,b):
    """
    x in a,b
    xright in b,b+a
    xleft in
    """
    xleft = 2.*a-x
    xright = 2.*b-x
    return np.concatenate((xleft,x,xright))
#    return np.pad(x,x.shape[0], 'reflect',reflect_type='odd')

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',action='store',type='string',default=None,help='Output folder', dest='output')
    parser.add_option('-d',action='store',type='string',default=None,help='data folder', dest='data')
    (options,args)=parser.parse_args()

    out_folder = options.output
    np.random.seed(69)
    all_files = os.listdir(options.data)
    events_list = [f for f in all_files if '.' not in f]

    omega_true = CosmologicalParameters(0.7,0.3,0.7)

    Nbins = 128
#    events_list = events_list[:50]
    cls = []
    cls_om = []
    h_joint_cls = []
    om_joint_cls = []
    h_cdfs = []
    h_pdfs = []
    om_cdfs = []
    all_posteriors = []
    redshift_posteriors = {}
    pool = mp.Pool(mp.cpu_count())

    x_flat = np.linspace(0.5,1.0,Nbins)
    y_flat = np.linspace(0.0,1.0,Nbins)

    joint_posterior = np.zeros((Nbins,Nbins),dtype=np.float64)-2.0*np.log(Nbins)
    
    dx = np.diff(x_flat)[0]
    dy = np.diff(y_flat)[0]
    X,Y = np.meshgrid(x_flat,y_flat)

    bound_l = 0.0
    bound_h = 1.0

    for k,e in enumerate(events_list):
        try:
            print "processing %s (%d/%d)"%(e,k+1,len(events_list))
            posteriors = np.genfromtxt(os.path.join(options.data,e+"/posterior.dat"),names=True)

            rvs = np.column_stack(
                                  (reflective_boundaries(posteriors['h'],x_flat.min(),x_flat.max()),
                                   reflective_boundaries(posteriors['om'],y_flat.min(),y_flat.max()))
                                  )

            cls.append(np.percentile(posteriors['h'],[5,50,95]))
            cls_om.append(np.percentile(posteriors['om'],[5,50,95]))
            model = initialise_dpgmm(2,rvs)
            logdensity = compute_dpgmm(model,max_sticks=8)
            logZ = evaluate_grid(logdensity,x_flat,y_flat)
            joint_posterior += logZ
            
            all_posteriors.append(joint_posterior)
            pickle.dump(joint_posterior,open(os.path.join(options.data,"joint_log_post_%d.p"%k),"w"))

            joint_posterior = renormalise(joint_posterior,dx,dy)
            single_posterior = renormalise(logZ,dx,dy)

            pickle.dump(logZ,open(os.path.join(options.data,"log_post_%d.p"%k),"w"))
            h_pdf = marginalise(joint_posterior,dy,1)
            single_h_pdf = marginalise(single_posterior,dy,1)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_flat,h_pdf,lw=2.0,color='k')
            ax.plot(x_flat,single_h_pdf,lw=1.0,color='r',alpha=0.5)
            ax.hist(posteriors['h'],bins=x_flat,alpha=0.5,normed=True,facecolor="0.9")
            ax.axvline(0.7,linestyle='dashed',color='k')
            ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
            ax.set_ylabel(r"$\mathrm{probability}$ $\mathrm{density}$",fontsize=18)
            ax.set_xlim(x_flat.min(),x_flat.max())
            fig.savefig(os.path.join(options.data,"h_%d.pdf"%k),bbox_inches='tight')

            plt.close(fig)
            om_pdf = marginalise(joint_posterior,dx,0)
            single_om_pdf = marginalise(single_posterior,dx,0)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(y_flat,om_pdf,lw=2.0,color='k')
            ax.plot(y_flat,single_om_pdf,lw=1.0,color='r',alpha=0.5)
            ax.hist(posteriors['om'],alpha=0.5,normed=True,bins=y_flat,facecolor="0.9")
            ax.axvline(0.3,linestyle='dashed',color='k')
            ax.set_xlabel(r"$\Omega_m$",fontsize=18)
            ax.set_ylabel(r"$\mathrm{probability}$ $\mathrm{density}$",fontsize=18)
#            ax.set_xlim(0.04,0.9)
            fig.savefig(os.path.join(options.data,"om_%d.pdf"%k),bbox_inches='tight')
            plt.close(fig)

            h_cdf = np.cumsum(np.sum(joint_posterior*dy,axis=1)*dx)
            om_cdf = np.cumsum(np.sum(joint_posterior*dx,axis=0)*dy)
            h_joint_cls.append([x_flat[np.abs(h_cdf - l).argmin()]+dx/2. for l in 0.05,0.50,0.95])
            om_joint_cls.append([y_flat[np.abs(om_cdf - l).argmin()]+dy/2. for l in 0.05,0.50,0.95])
            joint_posterior = np.log(joint_posterior)
            h_cdfs.append(h_cdf)
            h_pdfs.append(h_pdf)
        except:
            raise

    cls = np.array(cls)
    cls_om = np.array(cls_om)
    h_joint_cls = np.array(h_joint_cls)
    om_joint_cls = np.array(om_joint_cls)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    levs = np.sort(FindHeightForLevel(joint_posterior.T,[0.68,0.95]))
    C = ax.contour(X,Y,joint_posterior.T,levs,linewidths=2.0,colors='black')
#    ax.scatter(posteriors['h'],posteriors['om'],s=1,alpha=0.5)
    ax.grid(alpha=0.5,linestyle='dotted')
    ax.axvline(0.7,color='k',linestyle='dashed')
    ax.axhline(0.3,color='k',linestyle='dashed')
    ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
    ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    ax.set_ylim(0.1,1.0)
    plt.savefig(os.path.join(options.data,"joint_posterior.pdf"),bbox_inches='tight')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.arange(1,cls.shape[0]+1),cls[:,1],yerr=[cls[:,2]-cls[:,1],cls[:,1]-cls[:,0]],marker='o', fmt='o')
    ax.errorbar(np.arange(1,h_joint_cls.shape[0]+1),h_joint_cls[:,1],yerr=[h_joint_cls[:,2]-h_joint_cls[:,1],h_joint_cls[:,1]-h_joint_cls[:,0]],color='k',ecolor='k',marker='o', fmt='o')
    ax.set_xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{events}$",fontsize=18)
    ax.set_ylabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
    ax.axhline(0.73,color='k',linestyle='dashed')
    ax.grid(alpha=0.5)
    plt.savefig(os.path.join(options.data,"h_vs_n.pdf"),bbox_inches='tight')
    np.savetxt(os.path.join(options.data,"h_confidence_levels.txt"),h_joint_cls)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.arange(1,cls_om.shape[0]+1),cls_om[:,1],yerr=[cls_om[:,2]-cls_om[:,1],cls_om[:,1]-cls_om[:,0]],marker='o', fmt='o')
    ax.errorbar(np.arange(1,om_joint_cls.shape[0]+1),om_joint_cls[:,1],yerr=[om_joint_cls[:,2]-om_joint_cls[:,1],om_joint_cls[:,1]-om_joint_cls[:,0]],color='k',ecolor='k',marker='o', fmt='o')
    ax.set_xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{events}$",fontsize=18)
    ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    ax.grid(alpha=0.5)
    ax.axhline(0.3,color='k',linestyle='dashed')
    plt.savefig(os.path.join(options.data,"om_vs_n.pdf"),bbox_inches='tight')
    np.savetxt(os.path.join(options.data,"om_confidence_levels.txt"),om_joint_cls)
