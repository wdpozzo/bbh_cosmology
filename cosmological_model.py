#!/usr/bin/env python
import unittest
import numpy as np
import cpnest.model
import sys
import os
from optparse import OptionParser
import itertools as it
import cosmology as cs
import readdata
from scipy.misc import logsumexp

class CosmologicalModel(cpnest.model.Model):

    names=['h','om']
    bounds=[[0.5,1.0],[0.0,1.0]]
    
    def __init__(self, ID=None, threshold=None, pdf = None,**kwargs):
        super(CosmologicalModel,self).__init__(**kwargs)
        # Set up the data
        self.id = ID
        self.pdf = pdf
        self.galaxy_catalog=np.loadtxt('galaxy_catalogs/galaxy_catalog_threshold_%d_%04d.txt'%(threshold,self.id))
        self.N = len(self.galaxy_catalog)
        self.logN = np.log(len(self.galaxy_catalog))

        print('Event id {0} --> {1} galaxies'.format(self.id,self.N))

    def log_prior(self,x):
        return super(CosmologicalModel,self).log_prior(x)

    def log_likelihood(self,x):
        O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'])
        logL = cs.log_likelihood_single(self.pdf,self.galaxy_catalog,O)-self.logN
        return logL

usage=""" %prog (options)"""

if __name__=='__main__':
    parser=OptionParser(usage)
    parser.add_option('-o','--out-dir',default=None,type='string',metavar='DIR',help='Directory for output')
    parser.add_option('-t','--threads',default=None,type='int',metavar='N',help='Number of threads (default = 1/core)')
    parser.add_option('-d','--data',default=None,type='string',metavar='data',help='DPGMM data location')
    parser.add_option('-e','--event',default=1,type='int',metavar='event',help='event number')
    parser.add_option('-s','--threshold',default=20,type='float',metavar='threshold',help='telescope detection threshold')
    (opts,args)=parser.parse_args()
    
    pdfs,ids = readdata.find_events(opts.data)
    k = ids.index(opts.event)
    id = ids[k]
    p = pdfs[k]
    pdf = readdata.load_dpgmm_data([os.path.join(opts.data,'%s'%p)])[0]
    output_folder = os.path.join(opts.out_dir,str(id))
    
    # check if the requested catalog exists
    try:
        f = open ('galaxy_catalogs/galaxy_catalog_threshold_%d_%04d.txt'%(opts.threshold,id),'r')
    except:
        print("Galaxy catalog for event {0} not found! Generating".format(id))
        from generate_galaxy_catalog import *
        np.seterr(divide='ignore', invalid='ignore')
        sample_dpgmm(pdf, id, output = 'galaxy_catalogs/', threshold = opts.threshold, debug = False)

    work=cpnest.CPNest(CosmologicalModel(ID=id, threshold=opts.threshold, pdf = pdf),
                   verbose=2,
                   Poolsize=256,
                   Nthreads=opts.threads,
                   Nlive=1024,
                   maxmcmc=100,
                   output=output_folder)

    work.run()
    print('Evidence {0}'.format(work.NS.logZ))
    x = work.posterior_samples.ravel()

    import corner
    samps = np.column_stack((x['h'],x['om']))
    fig = corner.corner(samps,
           labels= [r'$h$',
                    r'$\Omega_m$'],
           quantiles=[0.05, 0.5, 0.95],
           show_titles=True, title_kwargs={"fontsize": 12},
           use_math_text=True, truths=[0.7,0.3],
           filename=os.path.join(output_folder,'joint_posterior.pdf'))
    fig.savefig(os.path.join(output_folder,'joint_posterior.pdf'))

    # copy the used galaxy_catalog in the output folder
    os.system("cp galaxy_catalogs/galaxy_catalog_threshold_%d_%04d.txt %s"%(opts.threshold,id,output_folder))
