import math, os, sys, re
import numpy as np
import mpmath as mp
import libstempo
import copy

import pylab as pl

from libstempo import GWB

day = 24 * 3600
year = 365.25 * day
DMk = 4.15e3           # Units MHz^2 cm^3 pc sec

def add_gwb(psr,dist=1,ngw=1000,seed=None,flow=1e-8,fhigh=1e-5,gwAmp=1e-20,alpha=-0.66,logspacing=True):
    """Add a stochastic background from inspiraling binaries, using the tempo2
    code that underlies the GWbkgrd plugin.

    Here 'dist' is the pulsar distance [in kpc]; 'ngw' is the number of binaries,
    'seed' (a negative integer) reseeds the GWbkgrd pseudorandom-number-generator,
    'flow' and 'fhigh' [Hz] determine the background band, 'gwAmp' and 'alpha'
    determine its amplitude and exponent, and setting 'logspacing' to False
    will use linear spacing for the individual sources.

    It is also possible to create a background object with

    gwb = GWB(ngw,seed,flow,fhigh,gwAmp,alpha,logspacing)

    then call the method gwb.add_gwb(pulsar[i],dist) repeatedly to get a
    consistent background for multiple pulsars.

    Returns the GWB object
    """
    

    gwb = GWB(ngw,seed,flow,fhigh,gwAmp,alpha,logspacing)
    gwb.add_gwb(psr,dist)

    return gwb

def add_dipole_gwb(psr,dist=1,ngw=1000,seed=None,flow=1e-8,fhigh=1e-5,gwAmp=1e-20, alpha=-0.66, \
        logspacing=True, dipoleamps=None, dipoledir=None, dipolemag=None):
        
    """Add a stochastic background from inspiraling binaries distributed
    according to a pure dipole distribution, using the tempo2
    code that underlies the GWdipolebkgrd plugin.

    The basic use is identical to that of 'add_gwb':
    Here 'dist' is the pulsar distance [in kpc]; 'ngw' is the number of binaries,
    'seed' (a negative integer) reseeds the GWbkgrd pseudorandom-number-generator,
    'flow' and 'fhigh' [Hz] determine the background band, 'gwAmp' and 'alpha'
    determine its amplitude and exponent, and setting 'logspacing' to False
    will use linear spacing for the individual sources.

    Additionally, the dipole component can be specified by using one of two
    methods:
    1) Specify the dipole direction as three dipole amplitudes, in the vector
    dipoleamps
    2) Specify the direction of the dipole as a magnitude dipolemag, and a vector
    dipoledir=[dipolephi, dipoletheta]

    It is also possible to create a background object with
    
    gwb = GWB(ngw,seed,flow,fhigh,gwAmp,alpha,logspacing)

    then call the method gwb.add_gwb(pulsar[i],dist) repeatedly to get a
    consistent background for multiple pulsars.
    
    Returns the GWB object
    """

    gwb = GWB(ngw,seed,flow,fhigh,gwAmp,alpha,logspacing,dipoleamps,dipoledir,dipolemag)
    gwb.add_gwb(psr,dist)
    
    return gwb

def _geti(x,i):
    return x[i] if isinstance(x,(tuple,list,np.ndarray)) else x

def fakepulsar(parfile,obstimes,toaerr,freq=1440.0,observatory='AXIS',flags=''):
    """Returns a libstempo tempopulsar object corresponding to a noiseless set
    of observations for the pulsar specified in 'parfile', with observations
    happening at times (MJD) given in the array (or list) 'obstimes', with
    measurement errors given by toaerr (us).

    A new timfile can then be saved with pulsar.savetim(). Re the other parameters:
    - 'toaerr' needs to be either a common error, or a list of errors
       of the same length of 'obstimes';
    - 'freq' can be either a common observation frequency in MHz, or a list;
       it defaults to 1440;
    - 'observatory' can be either a common observatory name, or a list;
       it defaults to the IPTA MDC 'AXIS';
    - 'flags' can be a string (such as '-sys EFF.EBPP.1360') or a list of strings;
       it defaults to an empty string."""

    import tempfile
    outfile = tempfile.NamedTemporaryFile(delete=False)

    outfile.write('FORMAT 1\n')
    outfile.write('MODE 1\n')

    obsname = 'fake_' + os.path.basename(parfile)
    if obsname[-4:] == '.par':
        obsname = obsname[:-4]

    for i,t in enumerate(obstimes):
        outfile.write('{0} {1} {2} {3} {4} {5}\n'.format(
            obsname,_geti(freq,i),t,_geti(toaerr,i),_geti(observatory,i),_geti(flags,i)
        ))

    timfile = outfile.name
    outfile.close()

    pulsar = libstempo.tempopulsar(parfile,timfile,dofit=False)
    pulsar.stoas[:] -= pulsar.residuals(updatebats=False) / 86400.0

    os.remove(timfile)

    return pulsar

def make_ideal(psr):
    """Adjust the TOAs so that the residuals to zero, then refit."""
    
    psr.stoas[:] -= psr.residuals() / 86400.0
    psr.fit()

def add_efac(psr,efac=1.0,seed=None):
    """Add nominal TOA errors, multiplied by `efac` factor.
    Optionally take a pseudorandom-number-generator seed."""
    
    if seed is not None:
        np.random.seed(seed)

    psr.stoas[:] += efac * psr.toaerrs * (1e-6 / day) * np.random.randn(psr.nobs)

def add_equad(psr,equad,seed=None):
    """Add quadrature noise of rms `equad` [s].
    Optionally take a pseudorandom-number-generator seed."""

    if seed is not None:
        np.random.seed(seed)
    
    psr.stoas[:] += (equad / day) * np.random.randn(psr.nobs)

def quantize(times,dt=1):
    bins    = np.arange(np.min(times),np.max(times)+dt,dt)
    indices = np.digitize(times,bins) # indices are labeled by "right edge"
    counts  = np.bincount(indices,minlength=len(bins)+1)

    bign, smalln = len(times), np.sum(counts > 0)

    t = np.zeros(smalln,'d')
    U = np.zeros((bign,smalln),'d')

    j = 0
    for i,c in enumerate(counts):
        if c > 0:
            U[indices == i,j] = 1
            t[j] = np.mean(times[indices == i])
            j = j + 1
    
    return t, U

def quantize_fast(times,dt=1):
    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    return t, U

# check that the two versions match
# t, U = quantize(np.array(psr.toas(),'d'),dt=1)
# t2, U2 = quantize_fast(np.array(psr.toas(),'d'),dt=1)
# print np.sum((t - t2)**2), np.all(U == U2)

def add_jitter(psr,equad,coarsegrain=0.1,seed=None):
    """Add correlated quadrature noise of rms `equad` [s],
    with coarse-graining time `coarsegrain` [days].
    Optionally take a pseudorandom-number-generator seed."""
    
    if seed is not None:
        np.random.seed(seed)

    t, U = quantize_fast(np.array(psr.toas(),'d'),0.1)
    psr.stoas[:] += (equad / day) * np.dot(U,np.random.randn(U.shape[1]))


def add_rednoise(psr,A,gamma,components=10,seed=None):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""
    
    noise2add = rednoise(psr.toas(),psr.nobs,A,gamma,components=components,seed=None)
    psr.stoas[:] += noise2add


def rednoise(t,nobs,A,gamma,components=10,seed=None,debug=False):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""
    
    if seed is not None:
        np.random.seed(seed)
    
    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)  # dates from  0 to 1 
    T = (day/year) * (maxx - minx)  # duration in years. 1/T : size of frequency bin
    
    size = 2*components
    F = np.zeros((nobs,size),'d')  # table of cosinus and sinus for each data
    f = np.zeros(size,'d')         # frequency of each component in yr^-1 for cos and sin
    
    for i in range(components):
        F[:,2*i]   = np.cos(2*math.pi*(i+1)*x)
        F[:,2*i+1] = np.sin(2*math.pi*(i+1)*x)
        
        f[2*i] = f[2*i+1] = (i+1) / T # frequency of each component in yr^-1 for cos and sin

    norm = A**2 * year**2 / (12 * math.pi**2 * T)
    prior = norm * f**(-gamma)
    
    y = np.sqrt(prior) * np.random.randn(size)
    added_noise = (1.0/day) * np.dot(F,y)

    if debug :
        print "min =",minx, " max =", maxx," T =",T," A =",A," norm =",norm," A^2*yr^2 =",A**2*year**2," A^2*yr^2/(12pi^2) =",A**2*year**2/(12*math.pi**2)
        fOut = open('Tmp_RN_freq.txt','w')
        fOut.write('#f(yr^-1) prior y\n')
        for i in range(components):
            fOut.write(repr(f[i])+" "+repr(prior[i])+" "+repr(y[i])+"\n")
        fOut.close()

        fOut = open('Tmp_RN_time.txt','w')
        fOut.write('#t RN\n')
        for i in range(nobs):
            fOut.write(repr(t[i])+" "+repr(added_noise[i])+"\n")
        fOut.close()

    return added_noise


def add_dm(psr,A,gamma,components=10,seed=None):
    """Add DM variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""

    noise2add = dm(psr.toas(),psr.nobs,psr.freqs,A,gamma,components=components,seed=None)
    psr.stoas[:] += noise2add
    

def dm(t,nobs,freqs,A,gamma,components=10,seed=None):
    """Add DM variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""
    
    if seed is not None:
        np.random.seed(seed)
    
    v = DMk / freqs**2
    
    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day/year) * (maxx - minx)
    
    size = 2*components
    F = np.zeros((nobs,size),'d')
    f = np.zeros(size,'d')
    
    for i in range(components):
        F[:,2*i]   = np.cos(2*math.pi*(i+1)*x)
        F[:,2*i+1] = np.sin(2*math.pi*(i+1)*x)
        
        f[2*i] = f[2*i+1] = (i+1) / T
    
    norm = A**2 * year**2 / (12 * math.pi**2 * T)
    prior = norm * f**(-gamma)
    
    y = np.sqrt(prior) * np.random.randn(size)
    added_noise = (1.0/day) * v * np.dot(F,y)

    return added_noise


def add_line(psr,f,A,offset=0.5):
    """Add a line of frequency `f` [Hz] and amplitude `A` [s],
    with origin at a fraction `offset` through the dataset."""
    
    t = psr.toas()
    t0 = offset * (np.max(t) - np.min(t))
    sine = A * np.cos(2 * math.pi * f * day * (t - t0))

    psr.stoas[:] += sine / day


def run(command, disp=False, NoExit=False):
    commandline = command % globals()
    if disp : 
        print "----> %s" % commandline
    
    try:
        assert(os.system(commandline) == 0)
    except:
        print 'Script %s failed at command "%s".' % (sys.argv[0],commandline)
        if not NoExit :
            sys.exit(1)


################ Class toas toasim
class toasim:

    def __init__(self,parfile,timfile=None,refpsr=None,verbose=0):
        """
        used to simulate a pulsar from a reference pulsar defined by its parfile OR by a libstempo.tempopulsar .
        The TOAs are created from existing TOAs (timfile or tempopulsar).
        """
        
        self.niter = 5
        self.prec = 25
        mp.mp.dps = self.prec
        self.verbose = verbose
        
        #### Checking parfile
        if not os.path.isfile(parfile):
            print "ERROR : No such file",parfile
            sys.exit(1)

        #### Checking tempopulsar and timfile. If it's a libstempo.tempopulsar create a local tim file 
        if refpsr!=None :
            timfile = "Tmp.tim"
            refpsr.savetim(timfile)
        else:
            if not os.path.isfile(timfile):
                print "ERROR : No such file",timfile
                sys.exit(1)
            refpsr = libstempo.tempopulsar(parfile=parfile,timfile=timfile,dofit=False)

        #### Extracting list of backend
        self.syslist = refpsr.listsys()
        self.psrname = refpsr.name
        if verbose!=0 : print "Pulsar :", self.psrname
        if verbose!=0 : print "Backend :",self.syslist

        #### Extracting EFAC, EQUAD, RN and DM from TN parameters in the parfile and remove them from the parfile
        if verbose!=0 : print "EFAC, EQUAD, RN and DM read in parfile",parfile
        self.EFAC  = np.ones(len(self.syslist))
        self.EQUAD = np.zeros(len(self.syslist)) 
        self.RNAmp = 0.
        self.RNGam = 1.1
        self.DMAmp = 0.
        self.DMGam = 1.1
        fIn = open(parfile,'r')
        lines = fIn.readlines()
        fIn.close()
        self.parlines = []
        self.parlinesTNEFEEQ = []
        self.parlinesTNRNDM = []
        for line in lines:
            w = re.split("\s+",line)
            if w[0]=="TNEF" or w[0]=="TNEQ" :
                self.isTNEFEQ = True
                iSys = -1
                for i in xrange(len(self.syslist)):
                    if self.syslist[i]==w[2]:
                        iSys = i
                if iSys == -1 :
                    print "ERROR : Load TNEF and TNEQ : the sys",w[2],"from",w,"has not been found in",self.syslist
                    sys.exit(1)
                if w[0]=="TNEF":
                    self.EFAC[iSys] = float(w[3])
                if w[0]=="TNEQ":
                    self.EQUAD[iSys] = 10**float(w[3])
            
            if w[0]=="TNRedAmp" :
                self.RNAmp = 10**float(w[1])
            if w[0]=="TNRedGam" :
                self.RNGam = float(w[1]) 
            if w[0]=="TNDMAmp" :
                self.DMAmp = 10**float(w[1])
            if w[0]=="TNDMGam" :
                self.DMGam = float(w[1]) 
                    
            if w[0]=="TNEF" or w[0]=="TNEQ":
                self.parlinesTNEFEEQ.append(line)
            elif w[0]=="TNRedAmp" or w[0]=="TNRedGam" or w[0]=="TNDMAmp" or w[0]=="TNDMGam" :
                self.parlinesTNRNDM.append(line)
            else:
                self.parlines.append(line)
            
        if verbose!=0 :
            for i in xrange(len(self.syslist)):
                print "\t -",self.syslist[i],": EFAC =",self.EFAC[i]," EQUAD =",self.EQUAD[i]
            print "\t - Red noise : amplitude =",self.RNAmp," gamma =",self.RNGam
            print "\t - DM : amplitude =",self.DMAmp," gamma =",self.DMGam
                
                
        #### Extracting list of toas and timlines :
        # The data are store in timTOAs, each element containing : 
        # toas in mpmath format, errs in mpmath format, sys, words of original line, observation frequency
        self.timHead = []
        self.timTOAs = []   
        self.loadtim(timfile)
                
        #### Idealizing each backend
        for xSys in self.syslist :
            if verbose!=0 : print "Idealizing backend",xSys,"..."
            localverbose = 0
            if verbose>1 : localverbose = 1
            for ite in xrange(self.niter):
                ### Using libstempo.tempopulsar to get residuals
                self.savepar("TmpSys", verbose=localverbose)
                self.savetim("TmpSys", xSys, verbose=localverbose)
                psrS =  libstempo.tempopulsar(parfile="TmpSys.par",timfile="TmpSys.tim",dofit=False)
                res = psrS.residuals()
                ### Substract residuals
                k=0
                for itoa in xrange(len(self.timTOAs)) :
                    if self.timTOAs[itoa][2]==xSys :
                        self.timTOAs[itoa][0] -= mp.mpf(repr(res[k])) / mp.mpf('86400.')
                        k += 1
                ### Substract residuals to libstempo.tempopulsar just to compute RMS
                if verbose>1 : print "\t - iteration %d : before : rms = %e" % ( ite, psrS.rms())
                psrS.stoas[:] -= res / 86400.
                if verbose>1 : print "\t - iteration %d : after : rms = %e , residuals mean (std) = %e (%e) s" % ( ite, psrS.rms(), np.mean(res), np.std(res) )
                
        #### Keep ideal TOAs in memory to be the basis of 
        self.timTOAsIdeal = copy.deepcopy(self.timTOAs)

    
        #### TODO 
        # In the idealized data, there are still jumps thus it seems that the jumps are different from the original one :
        # - Why ?
        # - Do we need to fit for the jumps now and update the jumps value ? 
    
    

    def loadtim(self, timfile, InT2psr=True, verbose=-1):
        """ Load TOAs from timfile in TOAs """
        if verbose==-1 : verbose = self.verbose
        if verbose!=0 : print "Load toas from",timfile,"..."
        
        if not os.path.isfile(timfile):
            print "ERROR : No such file",timfile
            sys.exit(1)
        fIn = open(timfile,'r')
        lines = fIn.readlines()
        fIn.close()
        ReadHead = False
        if len(self.timHead)==0:
            ReadHead = True
        for line in lines :
            w = re.split("\s+",line)
            if w[0]=="" :
                w = w[1:]
            if ReadHead and (w[0]=="FORMAT" or w[0]=="MODE") :
                self.timHead.append(w)
            if w[0]=="INCLUDE":   
                ### Manage nested tim files
                if w[1][0]=="/":
                    self.loadtim(w[1])
                else:
                    self.loadtim(os.path.dirname(timfile)+"/"+w[1])
            elif len(w)>=4 and w[0][0]!='C' and w[0]!="FORMAT" and w[0]!="MODE":
                ### Read TOA line
                ReadHead = False
                ## Extract name of backend
                lsys=''
                for iw in xrange(len(w)) :
                    if w[iw]=='-sys':
                        lsys = w[iw+1]
                if lsys=='':
                    lsys = 'default'
                self.timTOAs.append([mp.mpf(w[2]),mp.mpf(w[3]),lsys,w,float(w[1])])



    def savetim(self, basename, sysname='all', multifile=False, IdealTOAs=False, verbose=-1):
        "Save TOA in tim file. It is strictly the original tim, except the column 3 and 4 corresponding to the new toa and error."
        if verbose==-1 : verbose = self.verbose
        if verbose!=0 : print "Save TOAs in",basename+".tim ..."
        
        fOut = open(basename+".tim",'w')
        for line in self.timHead :
            for iw in xrange(len(line)):
                if iw!=0:
                    fOut.write(" ")
                fOut.write(line[iw])
            fOut.write("\n")
        if multifile :
            os.system("mkdir -p "+basename)
            for xsys in self.syslist :
                fOut.write("INCLUDE "+os.path.basename(basename)+"/"+xsys+".tim\n")
                self.savetim(basename+"/"+xsys, sysname=xsys, multifile=False, IdealTOAs=IdealTOAs)
        else :
            tTOAs = self.timTOAs
            if IdealTOAs :
                tTOAs = self.timTOAsIdeal
            for tl in tTOAs :
                if sysname=='all' or sysname==tl[2] :
                    tl[3][2] = "%s" % (mp.nstr(tl[0],n=self.prec)) # update the word corresponding to the TOAs
                    tl[3][3] = "%s" % (mp.nstr(tl[1],n=self.prec)) # update the word corresponding to the errors 
                    ## Write tim line as a series of words
                    for iw in xrange(len(tl[3])):
                        if iw!=0 :
                            fOut.write(" ")
                        fOut.write(tl[3][iw])
                    fOut.write("\n")
        fOut.close()


    def savepar(self, basename, AddTNEFEQ=False, AddTNRNDM=False, verbose=-1):
        "Save parameters in par file. It's exactly the original par file except that TN parameters can be added or not."
        if verbose==-1 : verbose = self.verbose
        if verbose==-1 : verbose = self.verbose
        
        if verbose!=0 : print "Save parameters in",basename+".par ..."
        fIn = open(basename+".par",'w')
        for line in self.parlines:
            fIn.write(line)
        if AddTNEFEQ:
            for line in self.parlinesTNEFEEQ:
                fIn.write(line)
        if AddTNRNDM:
            for line in self.parlinesTNRNDM:
                fIn.write(line)
        fIn.close()
        

    def add_whitenoise(self, efac=1.0, equad=0.0, sysname='all', seed=None, verbose=-1):
        """Add nominal TOA errors, multiplied by `efac` factor and adding an equad.
        Optionally take a pseudorandom-number-generator seed.
        The errorbar in the tim file after this operation will contain the efac and equad """
        if verbose==-1 : verbose = self.verbose
        if verbose!=0 : print "Add white noise on sys","with EFAC =",efac," and EQUAD =",equad,"( seed =",seed,") ..."
    
        if seed is not None:
            np.random.seed(seed)
        
        for toa in self.timTOAs :
            if toa[2]==sysname or sysname=='all' :
                toa[1] = mp.sqrt ( mp.mpf(equad*equad*1e12) + toa[1]*toa[1]*mp.mpf(efac*efac) )
                toa[0] += toa[1] * mp.mpf((1e-6 / day) * np.random.randn())


    def add_whitenoiseTN(self, efacGlobal=1.0, equadGlobal=0.0, seed=None, verbose=-1):
        """Add nominal TOA errors, multiplied by efac factor and adding quadratically an equad.
        The used equad and efac are the TNEF and TNEQ parameters in the parfile.
        Optionally take a pseudorandom-number-generator seed.
        The errorbar in the tim file after this operation will NOT contain the `efac` and `equad` 
        but the TN parameter will be added in the parfile"""
        if verbose==-1 : verbose = self.verbose
        if verbose!=0 : print "Add white noise from TN parameters and global EFAC =",efacGlobal," and EQUAD =",equadGlobal," ( seed =",seed,") ..."
        
        if seed is not None:
            np.random.seed(seed)
        
        if len(self.parlinesTNEFEEQ)==0:
            print "WARNING : there is no TNEF and TNEQ in the original parfile so individual EFAC is 1.0 and individual EQUAD is 0.0 "
        for i in xrange(len(self.syslist)):
            xsys = self.syslist[i]
            efac = self.EFAC[i]*efacGlobal
            equad = self.EQUAD[i]
            self.parlines.append("TNEF -sys "+xsys+" "+repr(efac)+"\n")
            self.parlines.append("TNEQ -sys "+xsys+" "+repr(np.log10(equad))+"\n")
            for toa in self.timTOAs :
                if toa[2]== xsys:
                    err = mp.sqrt (  mp.mpf(equadGlobal*equadGlobal*1e12) + mp.mpf(equad*equad*1e12) + toa[1]*toa[1]*mp.mpf(efac*efac) )
                    toa[0] += err * mp.mpf((1e-6 / day) * np.random.randn())


    def add_rednoise(self,A,gamma,components=50,seed=None,sysname="all", verbose=-1):
        """Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
        using `components` Fourier bases.
        Optionally take a pseudorandom-number-generator seed.
        Use directly the simple function.
        With sysname, it is possible to add red noise only on one backend."""
        if verbose==-1 : verbose = self.verbose

        t = []
        #### Create the TOAs array for the specified sysname using ideal data as reference
        ## TODO : check that is correct to use ideal data as reference 
        for itoa in xrange(len(self.timTOAsIdeal)):
            if sysname==self.timTOAsIdeal[itoa][2] or sysname=="all":
                t.append(np.float128(self.timTOAsIdeal[itoa][0]))
        #### Create red noise data
        if verbose!=0 : print "Add red noise with amplitude =",A,", gamma =", gamma, "and using",components,"components ( seed =",seed,") ..."
        RNval = rednoise(np.array(t),len(t),A,gamma,components=components,seed=seed,debug=False)
        itt=0
        #### Add rednoise data to the TOAs
        for itoa in xrange(len(self.timTOAs)) :
            if sysname==self.timTOAs[2] or sysname=="all":
                self.timTOAs[itoa][0] += mp.mpf(RNval[itt])
                itt += 1


    def add_rednoiseTN(self,components=50,seed=None,sysname="all", verbose=-1):
        """Add red noise using the TN value read in the original par file."""
        if verbose==-1 : verbose = self.verbose
        
        if len(self.parlinesTNRNDM)==0:
            print "WARNING : No addded red noise because there is no TNRed in the original parfile"
            return False
        else :
            if verbose!=0 : print "Add red noise from TN parameters (amplitude =",self.RNAmp,"and gamma =", self.RNGam, ") using",components,"components ( seed =",seed,") ..."
            self.add_rednoise(self.RNAmp,self.RNGam,components=components,seed=seed,sysname=sysname)
            return True
        
                    
    def add_dm(self,A,gamma,components=50,seed=None,sysname="all", verbose=-1):
        """Add DM variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
        using `components` Fourier bases.
        Optionally take a pseudorandom-number-generator seed.
        WARNING: the amplitude definition is the same as for red noise, 
        so multiply by np.sqrt(12)*np.pi if temponest definition.
        """
        if verbose==-1 : verbose = self.verbose
        
        t = []
        f = []
        #### Create the TOAs array for the specified sysname using ideal data as reference
        ## TODO : check that is correct to use ideal data as reference 
        for itoa in xrange(len(self.timTOAsIdeal)):
            if sysname==self.timTOAsIdeal[itoa][2] or sysname=="all":
                t.append(np.float128(self.timTOAsIdeal[itoa][0]))
                f.append(np.float64(self.timTOAsIdeal[itoa][0]))
        #### Create red noise data
        if verbose!=0 : print "Add DM with amplitude =",A,", gamma =", gamma, "and using",components,"components ( seed =",seed,") ..."
        DMval = dm(np.array(t),len(t),np.array(f),A,gamma,components=components,seed=seed)
        itt=0
        #### Add rednoise data to the TOAs
        for itoa in xrange(len(self.timTOAs)) :
            if sysname==self.timTOAs[2] or sysname=="all":
                self.timTOAs[itoa][0] += mp.mpf(DMval[itt])
                itt += 1
            

    def add_dmTN(self,components=50,seed=None,sysname="all", verbose=-1):
        """Add red noise using the TN value read in the original par file."""
        if verbose==-1 : verbose = self.verbose
        
        if len(self.parlinesTNRNDM)==0:
            print "WARNING : No addded DM because there is no TNRed in the original parfile"
            return False
        else :
            if verbose!=0 : print "Add DM from TN parameters (amplitude =",self.DMAmp,"and gamma =", self.DMGam, ") using",components,"components ( seed =",seed,") ..."
            self.add_dm(self.DMAmp*np.sqrt(12)*np.pi,self.DMGam,components=components,seed=seed,sysname=sysname)
            return True


    def add_gwb(self, gwb, dist, InjectionFile="None", verbose=-1):
        """ Add GW background on simulated TOAs using a GWB object 
        from libstempo and the pulsar distance in kpc."""
        if verbose==-1 : verbose = self.verbose
        
        if verbose!=0 : print "Add GWB ..."
        #### Making libstempo.tempopulsar save in parfile and timfile
        localverbose=0
        if verbose>1 : localverbose=1
        self.savepar("TmpIdeal", verbose=localverbose)
        self.savetim("TmpIdeal",IdealTOAs=True, verbose=localverbose)
        psr = libstempo.tempopulsar(parfile="TmpIdeal.par",timfile="TmpIdeal.tim",dofit=False)
        #### Creating GWB data
        GWBval = gwb.add_gwb(psr,dist)
        #### Adding data
        fOut = None
        if InjectionFile!="None":
            fOut = open(InjectionFile,'w')
            fOut.write("#TOAIdeal GWB TOARelBeforeInjection TOARelAfterInjection DiffTOAAft-Bef\n")
        for itoa in xrange(len(self.timTOAs)) :
            TOABefInj = self.timTOAs[itoa][0]
            self.timTOAs[itoa][0] += mp.mpf(np.float64(GWBval[itoa]),n=self.prec)
            if fOut!=None :
                fOut.write(mp.nstr(self.timTOAsIdeal[itoa][0],n=self.prec)+" "+repr(GWBval[itoa])+" "\
                           +mp.nstr(TOABefInj-self.timTOAsIdeal[0][0],n=self.prec)+" "\
                           +mp.nstr(self.timTOAs[itoa][0]-self.timTOAsIdeal[0][0],n=self.prec)+" "\
                           +mp.nstr(self.timTOAs[itoa][0]-TOABefInj,n=self.prec)+"\n")
        if fOut!=None :
            fOut.close()



########### End of class toasim









