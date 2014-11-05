#!/usr/bin/env python

import os
import sys
import re
import string
import numpy as np
from math import pi
import pylab as pl

import libstempo as T
import libstempo.toasim as LT
from libstempo import GWB

################################ MAIN PROGRAM ################################

#RefDate=datetime.date(2014, 1, 1)
RefMJD_d=56658 # 01/01/2014


from optparse import OptionParser

parser = OptionParser(usage="usage: %prog [options] \n \
                      By default no noise are added. For adding noises, you have to specify where to read them, so :\n\
                      --UseTNEFEQ --UseTNRND : all noises are TN parameters from par files\n\
                      --UseNoiseFile : all noises are read in noise files JXXXX-XXXX-Noises.txt\n\
                      --UseTNEFEQ --UseNoiseFile : EFAC and EQUAD read in TN parameters and others noises from JXXXX-XXXX-Noises.txt",
                      version="04.11.14, version 1.2,  Antoine Petiteau ")  

#### General options
parser.add_option("-W", "--WhiteNoiseOnly",
                  action="store_true", dest="WhiteNoiseOnly", default=False,
                  help="Use white noise only [Default: false]")

parser.add_option("-i", "--InputDir",
                  type="string", dest="InputDir", default=".",
                  help="Input directory  [Default: current ]")

parser.add_option("-o", "--OutputDir",
                  type="string", dest="OutputDir", default="Simu",
                  help="Output directory  [Default: current ]")

parser.add_option("-p", "--ParExtention",
                  type="string", dest="ParExtention", default=".par",
                  help="Extention of the parfile. Example : .ML.par  [Default: .par ]")

parser.add_option("-c", "--Components",
                  type="int", dest="Components", default=50,
                  help="Number of red noise compnents [Default: 50 ]")

parser.add_option("-E", "--UseTNEFEQ",
                  action="store_true", dest="UseTNEFEQ", default=False,
                  help="Use EFAC and EQUAD from TN parameters read in parfile [default: False ]")

parser.add_option("-R", "--UseTNRNDM",
                  action="store_true", dest="UseTNRNDM", default=False,
                  help="Use Red Noise and DM from TN parameters read in parfile [default: False ]")

parser.add_option("-F", "--UseNoiseFile",
                  action="store_true", dest="UseNoiseFile", default=False,
                  help="Use noise in the noise file JXXXX-XXXX-Noises.txt [Default: false]")

parser.add_option("-P", "--Plots",
                  action="store_true", dest="Plots", default=False,
                  help="Make comparison plots [default: False ]")

parser.add_option("-I", "--SaveIdeal",
                  action="store_true", dest="SaveIdeal", default=False,
                  help="Save ideal TOAs in timfile in directory Ideal  [default: False ]")

parser.add_option("-G", "--GWB",
                  type="string", dest="GWB", default="None",
                  help="Add a GW background in the form Amp,Gam . Ex: 1.0e-14,4.3  [default: False ]")

parser.add_option("-g", "--SaveInjection",
                  action="store_true", dest="SaveInjection", default=False,
                  help="Save injection GWB data [default: False ]")

parser.add_option("-S", "--Seed",
                  type="int", dest="Seed", default=0,
                  help="Seed  [default: 0 --> random ]")

parser.add_option("-v", "--verbose",
                  type="int", dest="verbose", default=0,
                  help="Verbose level : 0, 1 or 2 [Default 0]")



(options, args) = parser.parse_args()


#if len(args) < 1:
#    parser.error("I need to know the ParTim directory  (containing -Noises.txt) and the data directory (containing pulsars.info) ")


DirIn   = options.InputDir
DirSimu = options.OutputDir+"/"
DirPlot = DirSimu+"/PlotCompare/"
DirIdeal = DirSimu+"/Ideal/"


if (not os.path.isdir(DirSimu)) : 
    os.mkdir(DirSimu)
if (not os.path.isdir(DirPlot)) : 
    os.mkdir(DirPlot)

if options.SaveIdeal and (not os.path.isdir(DirIdeal)) : 
    os.mkdir(DirIdeal)


listDir = os.listdir(DirIn)
listDir.sort()

psrs = [] # list of pulsar info : name, subdir, distance in kiloparsec
for xf in listDir :
    if xf[0]=="J" :
        if xf[-len(options.ParExtention):]==options.ParExtention :
            psrs.append([xf[0:-len(options.ParExtention)],"",-1.0])
        if os.path.isdir(DirIn+"/"+xf):
            psrs.append([xf,xf,1.0])
            listSubDir = os.listdir(DirIn+"/"+xf)
                    

RandSeed = int(np.random.rand()*1e6)
if options.Seed != 0:
    RandSeed = options.Seed


gwb = None
if options.GWB!="None" :
    ### Create GWB 
    w = re.split(",",options.GWB)
    gwb = GWB(ngw=10000,flow=1e-10,fhigh=1e-5,gwAmp=float(w[0]),alpha=(3.-float(w[1]))/2.) 

    ### Load distance
    distf = DirIn+"/distance.dat"
    if (not os.path.isfile(distf)):
        print "ERROR : No such file", distf
        sys.exit(1)
    fIn = open(distf,'r')
    lines = fIn.readlines()
    fIn.close()
    for i in xrange(len(psrs)):
        p = psrs[i][0]
        for line in lines:
            w = re.split("\s+",line)
            if p==w[0]:
                psrs[i][2] = float(w[1])
        if psrs[i][2] < 0. :
            print "ERROR : the distance of",p,"is not found in", distf
            sys.exit(1)


for pp in psrs :

    p = pp[0]
    subdir = pp[1]
    dist = pp[2]
    print "====================",p,"===================="
    
    NoWN = True # False if EFAC and EQUAD have already been added
    NoRN = True # False if Red Noise have already been added
    NoDM = True # False if DM have already been added
    
    parfile = p+options.ParExtention
    timfile = p+".tim"
    noifile = p+"-Noises.txt"
    DirInSub = DirIn+"/"+subdir
    if (not os.path.isfile(DirInSub+"/"+parfile)):
        print "ERROR : par-file", DirInSub+"/"+parfile, "is not found"
        sys.exit(1)
    if (not os.path.isfile(DirInSub+"/"+timfile)):
        timfile=p+"_all.tim"
        if (not os.path.isfile(DirInSub+"/"+timfile)):
            print "ERROR : tim-file", DirInSub+"/"+timfile, "is not found"
            sys.exit(1)
    
    
    RandSeed = RandSeed + 1


    ##### Making a simulated pulsar object starting with an ideal pulsar 
    if options.verbose!=0 : print ">>>>> Making an ideal pulsar ..."
    spsr = LT.toasim(parfile=DirInSub+"/"+parfile, timfile=DirInSub+"/"+timfile, verbose=options.verbose)
    
    
    ##### Add white noise from noise file
    RNamp = 0.0
    RNgam = 1.1
    DMamp = 0.0
    DMgam = 1.1
    EQg = 0.0
    EFg = 1.0
    EFEQs=[]
    if options.UseNoiseFile :
        
        noifile=p+"-Noises.txt"
        ##### Extracting noise 
        if options.verbose!=0 : print ">>>>> Reading noise file ..."
        
        if (not os.path.isfile(DirInSub+"/"+noifile)):
            print "ERROR : noise-file", DirInSub+"/"+noifile, "is not found"
            sys.exit(1)
        fIn=open(DirInSub+"/"+noifile,'r')
        for line in fIn:
            w=re.split("\s+",line)
            if w[1]=="Red" :
                if w[2]=="Amp" :
                    RNamp = 10**float(w[3])
                if w[2]=="Index" :
                    RNgam = float(w[3])
            if w[1]=="DM" :
                if w[2]=="Amp" :
                    DMamp = 10**float(w[3])
                if w[2]=="Index" :
                    DMgam = float(w[3])
            if w[1]=="EFAC" :
                if w[2]=="global" :
                    EFg = float(w[3])
                else:
                    iSys=-1
                    for i in xrange(len(EFEQs)):
                        if EFEQs[i][0] == w[2]:
                            iSys = i
                    if iSys!=-1:
                        EFEQs[iSys][1] =  float(w[3])
                    else :
                        EFEQs.append([w[2],float(w[3]),0.0])        
            if w[1]=="EQUAD" :
                if w[2]=="global" :
                    EQg = 10.**float(w[3])
                else:
                    iSys=-1
                    for i in xrange(len(EFEQs)):
                        if EFEQs[i][0] == w[2]:
                            iSys = i
                    if iSys!=-1:
                        EFEQs[iSys][2] = 10.**float(w[3])
                    else :
                        EFEQs.append([w[2],1.0,10.**float(w[3])])  
        fIn.close()
    
        if options.verbose!=0 : 
            print "\t - Red :",RNamp,RNgam
            print "\t - DM  :",DMamp,DMgam
            print "\t - EQg :",EQg
            print "\t - EFg :",EFg
            print "\t - EQEFs :",EFEQs

        #### Adding noises
        if RNamp!=-15. :
            spsr.add_rednoise(RNamp, RNgam, components=options.Components, seed=RandSeed+1)
            NoRN = False

        
        if DMamp!=-15. :
            spsr.add_dm(DMamp*np.sqrt(12)*np.pi, DMgam, components=options.Components, seed=RandSeed+2)
            NoDM = False

        NoWN = False
        if options.UseTNEFEQ :
            spsr.add_whitenoiseTN(efacGlobal=EFg, equadGlobal=EQg, seed=RandSeed+3)
        else:
            for x in EFEQs :
                spsr.add_whitenoise(efac=EFg*x[1], equad=np.sqrt(EQg**2+x[2]**2), sysname=x[0])
  

    ##### Add noises from TN parameters if not already done 
                                    
    if options.UseTNEFEQ and NoWN :
        spsr.add_whitenoiseTN(seed=RandSeed+4)
        NoWN = False
    
    if options.UseTNRNDM and NoRN :
        spsr.add_rednoiseTN(components=options.Components, seed=RandSeed+5)
        NoRN = False

    if options.UseTNRNDM and NoDM :
        spsr.add_dmTN(components=options.Components, seed=RandSeed+6)
        NoDM = False

    if gwb!=None :
        if options.SaveInjection :
            spsr.add_gwb(gwb,dist,InjectionFile=DirSimu+"/"+p+"_Inj.txt")
        else:
            spsr.add_gwb(gwb,dist)
    
    spsr.savepar(DirSimu+"/"+p, AddTNEFEQ=options.UseTNRNDM, AddTNRNDM=options.UseTNRNDM)
    spsr.savetim(DirSimu+"/"+p)
    if options.SaveIdeal :
        spsr.savetim(DirIdeal+"/"+p,IdealTOAs=True)
    

    ##### Recording new noise file
    fnoi=open(DirSimu+noifile,'w')
    if not options.UseTNRNDM and NoRN==False :
        fnoi.write(p+" Red Amp "+str(np.log10(RNamp))+" -1\n")
        fnoi.write(p+" Red Index "+str(RNgam)+" -1\n")
    if not options.UseTNRNDM and NoDM==False :
        fnoi.write(p+" DM Amp "+str(np.log10(DMamp))+" -1\n")
        fnoi.write(p+" DM Index "+str(DMgam)+" -1\n")
    if NoWN==False :
        fnoi.write(p+" EFAC global "+str(EFg)+" -1\n")
        fnoi.write(p+" EQUAD global "+str(EQg)+" -1\n")
        if not options.UseTNEFEQ :
            for x in EFEQs :
                fnoi.write(p+" EFAC "+x[0]+" "+str(x[1])+" -1\n")
                fnoi.write(p+" EQUAD "+x[0]+" "+str(x[2])+" -1\n")
    fnoi.close()


    ##### Checking plots

    if options.Plots :
        Npsrfits = 10

        ### Loading the original pulsar
        psr_o = T.tempopulsar(parfile=DirInSub+"/"+parfile,timfile=DirInSub+"/"+timfile)
        for i in xrange(Npsrfits):
            psr_o.fit()
        toas_d_o = psr_o.toas() # days
        res_o  = 1.0e6*psr_o.residuals() # microseconds
        errs_o = psr_o.toaerrs # microseconds
        toas_yr_o = 2014+(toas_d_o-RefMJD_d)/365.25
        rms_o = psr_o.rms()*1e6

        """
        ### Loading the simulated pulsar
        psr_n = T.tempopulsar(parfile=DirSimu+"/"+p+".par",timfile=DirSimu+"/"+p+".tim",dofit=False)
        toas_d_n = psr_n.toas() # days
        res_n  = 1.0e6*psr_n.residuals() # microseconds
        errs_n = psr_n.toaerrs # microseconds
        toas_yr_n = 2014+(toas_d_n-RefMJD_d)/365.25
        rms_n = psr_n.rms()*1e6 
        """
        
        ### Loading the simulated pulsar
        psr_nf = T.tempopulsar(parfile=DirSimu+"/"+p+".par",timfile=DirSimu+"/"+p+".tim",dofit=True)
        for i in xrange(Npsrfits):
            psr_nf.fit()
        toas_d_nf = psr_nf.toas() # days
        res_nf  = 1.0e6*psr_nf.residuals() # microseconds
        errs_nf = psr_nf.toaerrs # microseconds
        toas_yr_nf = 2014+(toas_d_nf-RefMJD_d)/365.25
        rms_nf = psr_nf.rms()*1e6 
        
        ### Making comparison plot
        pl.subplot(2,1,1)
        pl.errorbar(toas_yr_o,res_o,yerr=errs_o,fmt='+')
        pl.ylabel("residual (microsec)",size=10)
        pl.xticks(visible=False)
        pl.yticks(fontsize=9)
        pl.title("Original %s fitted (rms=%.3f microsec)"%(p,rms_o),size=11)
        pl.grid()

        """
        pl.subplot(3,1,2)
        #pl.errorbar(toas_yr_o,res_o,yerr=errs_o,fmt='r+',ecolor='red')
        pl.errorbar(toas_yr_n,res_n,yerr=errs_n,fmt='+')
        pl.ylabel("residual (microsec)",size=10)
        pl.xticks(visible=False)
        pl.yticks(fontsize=9)
        pl.title("New simulated %s not fitted (rms=%.3f microsec)"%(p,rms_n),size=11)
        pl.grid()
        """

        pl.subplot(2,1,2)
        #pl.errorbar(toas_yr_o,res_o,yerr=errs_o,fmt='r+',ecolor='red')
        pl.errorbar(toas_yr_nf,res_nf,yerr=errs_nf,fmt='+')
        pl.xlabel("date (years)",size=10)
        pl.ylabel("residual (microsec)",size=10)
        pl.xticks(fontsize=9)
        pl.yticks(fontsize=9)
        pl.title("New simulated %s fitted (rms=%.3f microsec)"%(p,rms_nf),size=11)
        pl.grid()

        pl.savefig(DirPlot+p+".png")
        pl.close()



            


