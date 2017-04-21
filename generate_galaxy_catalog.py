import numpy as np
import emcee
import readdata
import cosmology as cs
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import sys
import os
from optparse import OptionParser

def lnprior(theta, DistanceFunction, VolumeFunction):
    ra,dec,z,M = theta
    if 0.0 < z < 2.0 and 0.0 < ra < 2.0*np.pi and -np.pi/2. < dec < np.pi/2. and -30. < M < -10.:
        d = DistanceFunction(z)
#        M = absolute_magnitude(m,d)
        return np.log(VolumeFunction(z))+np.log(np.cos(dec))+np.log(SchecterFunction(M,Mstar,alpha,phistar))
    return -np.inf

def lnlike(theta, pdf, DistanceFunction):
    ra,dec,z,mu = theta
    dl = DistanceFunction(z)
    return logsumexp([prob.logL(cs.SphericalToCartesian(dl,ra,dec))+np.log(pdf[0][ind]) for ind,prob in enumerate(pdf[1])])

def lnprob(theta, pdf, DistanceFunction, VolumeFunction):
    lp = lnprior(theta, DistanceFunction, VolumeFunction)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, pdf, DistanceFunction)

def SchecterFunction(m,Mstar,alpha,phistar):
    return 0.4*np.log(10.0)*phistar*pow(10.0,-0.4*(alpha+1.0)*(m-Mstar))*np.exp(-pow(10,-0.4*(m-Mstar)))

def absolute_magnitude(m,d):
    return m-5.0*np.log10(1e5*d)

def apparent_magnitude(M,d):
    return M+5.0*np.log10(1e5*d)
"""
typical values for the r band (http://arxiv.org/abs/0806.4930)
"""
Mstar = -20.73 + 5.*np.log10(0.7)
alpha = -1.23
phistar = 0.009 * (0.7*0.7*0.7) #Mpc^3

def sample_dpgmm(pdf, id, output = None, threshold = 20, debug = False):
    # check if the full catalog exists already
    ndim = 4
    if os.path.isfile(os.path.join(output,'galaxy_catalog_%04d.txt'%id)):
        print "full catalog exists, not resampling"
        samples = np.loadtxt(os.path.join(output,'galaxy_catalog_%04d.txt'%id))
        (idy,) = np.where(samples[:,3] < threshold)
        x = np.array([samples[i,:] for i in idy])
        np.savetxt(os.path.join(output,'galaxy_catalog_threshold_%d_%04d.txt'%(threshold,id)),x)
    else:
    
        ndim, nwalkers = 4, 64
        CL_s = np.genfromtxt('confidence_levels/CL_%d.txt'%id)
        N = np.maximum(1,phistar*CL_s[6,1]) # we take 1 sigma volume

        width = 100
        nsteps = np.maximum(2*np.int(N/nwalkers),100)
        print "sampling %d galaxies. sampler initialised to do %d steps"%(N,nsteps)
        p0 = [[np.random.uniform(0.0,2.0*np.pi),
               np.random.uniform(-np.pi/2.,np.pi/2.),
               np.random.uniform(0.0,2.0),
               np.random.uniform(-30.0,-10.0)]
              for i in range(nwalkers)]
        O = cs.CosmologicalParameters(0.7,0.3,0.7)
        
        # make some interpolants for speed
        z = np.linspace(0.0,3.0,1000)
        dV = [O.ComovingVolumeElement(zi) for zi in z]
        VolumeInterpolant = interp1d(z,dV)
        Dl = [O.LuminosityDistance(zi) for zi in z]
        DistanceInterpolant = interp1d(z,Dl)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[pdf,DistanceInterpolant,VolumeInterpolant])
        
        for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
            n = int((width+1) * float(i) / nsteps)
            sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        sys.stdout.write("\n")
        samples = sampler.chain[:, nsteps/2:, :].reshape((-1, ndim))
        samples = np.vstack({tuple(row) for row in samples})

        d = DistanceInterpolant(samples[:,2])
        samples[:,3] = apparent_magnitude(samples[:,3],d)
        (idy,) = np.where(samples[:,3] < threshold)

        try:
            idx = np.random.choice(idy,replace=False,size=N)
        except:
            idx = idy

        x = np.array([samples[i,:] for i in idx])

        if output is None:
            os.system('mkdir -p galaxy_catalogs')
        else:
            os.system('mkdir -p %s'%output)
        np.savetxt(os.path.join(output,'galaxy_catalog_threshold_%d_%04d.txt'%(threshold,id)),x)
        np.savetxt(os.path.join(output,'galaxy_catalog_%04d.txt'%id),samples)
    if debug:
        import matplotlib.pyplot as pl
        for i in range(ndim):
            pl.figure()
            pl.hist(samples[::10,i], 100, color="k", histtype="step")
            pl.title("Dimension {0:d}".format(i))
        pl.figure()
        y = np.genfromtxt('Galaxies/galaxies_flat-%04d.txt'%id)
        pl.hist(samples[:,2],label='all',alpha=0.5,normed=True)
        pl.hist(y[:,2],label='old',alpha=0.5,normed=True)
        pl.hist(x[:,2],label='new',alpha=0.5,normed=True)
        pl.xlabel('redshift')
        pl.legend()
        pl.figure()
        pl.plot(samples[:,2],samples[:,3],'.r',alpha=0.5)
        pl.plot(samples[idy,2],samples[idy,3],'.b',alpha=0.5)
        pl.xlabel('redshift')
        pl.ylabel('absolute magnitude')
        pl.figure()
        pl.plot(samples[:,0],samples[:,1],'.r',alpha=0.5)
        pl.plot(samples[idy,0],samples[idy,1],'.b',alpha=0.5)
        pl.xlabel('right ascension')
        pl.ylabel('declination')
        pl.show()
    return samples

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    parser=OptionParser()
    parser.add_option('-o','--out-dir',default=None,type='string',metavar='DIR',help='Directory for output')
    parser.add_option('-d','--data',default=None,type='string',metavar='data',help='DPGMM data location')
    parser.add_option('-e','--event',default=1,type='int',metavar='event',help='event id')
    parser.add_option('-t','--threshold',default=20,type='float',metavar='threshold',help='telescope detection threshold')
    (opts,args)=parser.parse_args()

    pdfs,ids = readdata.find_events(opts.data)
    k = ids.index(opts.event)
    id = ids[k]
    p = pdfs[k]
    print "processing %s"%id
    pdf = readdata.load_dpgmm_data([os.path.join(opts.data,'%s'%p)])[0]
    sample_dpgmm(pdf, id, output = opts.out_dir, threshold = opts.threshold, debug = True)
