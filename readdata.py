import numpy as np
import sys
import os
from scipy.optimize import newton
import multiprocessing as mp
import cPickle as pickle
from dpgmm import *

def load_distance_data(filename):
    return np.loadtxt(filename,unpack = True)

def load_single_dpgmm_data(filename):
    return pickle.load(open(filename,'r'))

def load_dpgmm_data(filenames):
    return [load_single_dpgmm_data(f) for f in filenames]

def load_single_event_galaxy_data(filename):
    return np.loadtxt(filename)

def load_galaxy_catalog(filenames):
    return [load_single_event_galaxy_data(f) for f in filenames]

def inf_redshift(z,omega,dl):
    d = omega.LuminosityDistance(z)
    return dl - d

def find_redshift(omega, dl):
    return newton(inf_redshift,np.random.uniform(0.0,2.0),args=(omega,dl))

def sample_volume(omega,Vmax,zmin,zmax,N):
    samples = []
    pmax = omega.ComovingVolumeElement(zmax)/omega.ComovingVolume(zmax)
    while len(samples)<N:
        test = pmax*np.random.uniform(0.0,1.0)
        z = np.random.uniform(zmin,zmax)
        p = (omega.ComovingVolumeElement(z)/1e9/Vmax)
        if p > test:
            samples.append(z)
    return np.array(samples)

def redshift_extrema(d,e):
    # find the redshift limits for the given event
    zmins = []
    zmaxs = []
    for _ in range(1000):
        om = np.random.uniform(0.0,1.0)
        omega = cs.CosmologicalParameters(np.random.uniform(0.5,1.0),om,1.0-om)
        zmaxs.append(find_redshift(omega,d+e))
        zmins.append(find_redshift(omega,d-e))

    return np.min(zmins),np.max(zmaxs)

def find_events(location):
    all_files = os.listdir(location)
    pdfs = [f for f in all_files if 'DPGMM' in f]
    ids = [int((p.split('_')[-1]).split('.')[0]) for p in pdfs]
    return pdfs,ids

if __name__=="__main__":

    import cosmology as cs
    N = int(sys.argv[1])
    if N==None:
        print "Please specify the number of events you want to simulate"
        exit()
    O = cs.CosmologicalParameters(0.7,0.3,0.7)
    zmin = 0.0
    zmax = 0.05
    snr_threshold = 8.0
    reference_distance = 1000.0

    # generate sources sampled uniformly in comoving Volume
    Vmax = O.ComovingVolume(zmax)/1e9
    redshifts = sample_volume(O,Vmax,zmin,zmax,N)
    dl = np.array([O.LuminosityDistance(z) for z in redshifts])
    SNR = np.array([snr_threshold*reference_distance/d for d in dl])
    edl = dl*snr_threshold/SNR
    domega = (1000.0* 3.0462e-4)/SNR**2
    (detection,) = np.where(SNR > snr_threshold)
    np.savetxt('fake_data/distance.txt',np.column_stack((dl[detection],edl[detection])))

    D = dl[detection]
    E = edl[detection]
    for i in range(N):
        zmin,zmax = redshift_extrema(D[i],E[i])
        N_gals = np.maximum(1,np.int(1e-2*np.random.poisson(4.0*np.pi*domega[i]*E[i]**2)))
        z = sample_volume(O,Vmax,zmin,zmax,N_gals)
        np.savetxt('fake_data/redshift_%d.txt'%i,z)
