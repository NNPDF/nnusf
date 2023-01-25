#!/usr/bin/env python
# coding: utf-8

import os,sys
import lhapdf
import numpy as np
import matplotlib.pyplot as py
from matplotlib import gridspec
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
from pylab import *
import scipy
from scipy.integrate import dblquad

print("\n *********************************************************")
print("      Compute x dependence of NNUSF                         ")
print(" ***********************************************************\n")

#---------------------------------------------------------
#---------------------------------------------------------
#filelabel = "q14p0gev"
#filelabel = "q1gev"
#filelabel = "q0p5gev"
#filelabel = "q1p58gev"
filelabel = "q100gev"

if(filelabel=="q0p5gev"):
    q = 0.5 # GeV
if(filelabel=="q1gev"):
    q = 1 # GeV
if(filelabel=="q1p58gev"):
    q = 1.58 # GeV
if(filelabel=="q5p0gev"):
    q = 5 # GeV
if(filelabel=="q14p0gev"):
    q = 14 # GeV
if(filelabel=="q100gev"):
    q = 100 # GeV
 
#-------------------------------------------------------
#---------------------------------------------------------
# Plot settings
#---------------------------------------------------------
#---------------------------------------------------------

if(filelabel=="q0p5gev"):
    stringQ=r'$Q=0.5~{\rm GeV}$'
if(filelabel=="q1gev"):
    stringQ=r'$Q=1~{\rm GeV}$'
if(filelabel=="q1p58gev"):
    stringQ=r'$Q=1.6~{\rm GeV}$'
if(filelabel=="q5p0gev"):
    stringQ=r'$Q=5~{\rm GeV}$'
if(filelabel=="q14p0gev"):
    stringQ=r'$Q=14~{\rm GeV}$'
if(filelabel=="q100gev"):
    stringQ=r'$Q=100~{\rm GeV}$'

nx = 250
xmin = 1e-9
xmax= 9e-1
# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=1000
# number of flavours to be plotted
nfl=3
# Set x grid
X = np.logspace(log(xmin),log(xmax),nx)

# number of pdf sets
nset=3

nrep=np.zeros(nset, dtype='int')
nrep_max = 100

pdfset=["221110-A1-004","NNPDF31sx_nlo_as_0118_LHCb_nf_6","NNPDF40_nnlo_as_01180"]
fit1 = np.zeros((nrep_max,nfl,nx))
fit2 = np.zeros((nrep_max,nfl,nx))
fit3 = np.zeros((nrep_max,nfl,nx))

ids = np.array([1001,2001,3001])

for iset in range(nset):

    p=lhapdf.getPDFSet(pdfset[iset])
    nrep[iset]=int(p.get_entry("NumMembers"))-1
    print("nrep = ",nrep[iset])
#    if(nrep[iset] > nrep_max):
#        nrep[iset] = nrep_max
#    if(nrep[iset] > nrep_max):
#        print("Problem, too many replicas \n")
#        exit()
    #print(p.description)
#    print("nrep (fixed) = ",nrep[iset])

    if(iset==0):
        fit1 = np.zeros((nrep[iset],nfl,nx))
    if(iset==1):
        fit2 = np.zeros((nrep[iset],nfl,nx))
    if(iset==2):
        fit3 = np.zeros((nrep[iset],nfl,nx))

    # Run over replicas
    for i in range(1,nrep[iset]+1):
        p=lhapdf.mkPDF(pdfset[iset],i)
        lhapdf.setVerbosity(0)
        
        # Run over x arrat
        for k in range(nx):
            
            x = X[k]
            q2 = pow(q,2.0)

            # run over flavours
            for ifl in range(nfl):

                # For NNSUF
                if(iset==0):
                    fit1[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)

                # For hand calculation
                if(iset>0):
                    if(ifl==0): # F2^nu
                        if(iset==1):
                            fit2[i-1][ifl][k] =  ( p.xfxQ(-2,x,q) +  p.xfxQ(+1,x,q) + p.xfxQ(+3,x,q) + p.xfxQ(-4,x,q) + p.xfxQ(+5,x,q) )
                        if(iset==2):
                            fit3[i-1][ifl][k] = ( p.xfxQ(-2,x,q) +  p.xfxQ(+1,x,q) + p.xfxQ(+3,x,q) + p.xfxQ(-4,x,q) + p.xfxQ(+5,x,q)  )
                    #----------------------------------------------------------------
                    if(ifl==1): # F2^nubar
                        if(iset==1):
                            fit2[i-1][ifl][k] =  ( p.xfxQ(+2,x,q) +  p.xfxQ(-1,x,q) + p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) + p.xfxQ(-5,x,q) )
                        if(iset==2):
                            fit3[i-1][ifl][k] = ( p.xfxQ(+2,x,q) +  p.xfxQ(-1,x,q) + p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) + p.xfxQ(-5,x,q))
                    #----------------------------------------------------------------
                    if(ifl==2): # F2^nu+
                        if(iset==1):
                            fit2[i-1][ifl][k] = ( p.xfxQ(+1,x,q)+p.xfxQ(-1,x,q)+\
                                                  p.xfxQ(+2,x,q)+p.xfxQ(-2,x,q)+\
                                                  p.xfxQ(+3,x,q)+p.xfxQ(-3,x,q)+\
                                                  p.xfxQ(+4,x,q)+p.xfxQ(-4,x,q)+\
                                                  p.xfxQ(+5,x,q)+p.xfxQ(-5,x,q)) 
                        if(iset==2):
                            fit3[i-1][ifl][k] = ( p.xfxQ(+1,x,q)+p.xfxQ(-1,x,q)+\
                                                  p.xfxQ(+2,x,q)+p.xfxQ(-2,x,q)+\
                                                  p.xfxQ(+3,x,q)+p.xfxQ(-3,x,q)+\
                                                  p.xfxQ(+4,x,q)+p.xfxQ(-4,x,q) +\
                                                  p.xfxQ(+5,x,q)+p.xfxQ(-5,x,q) ) 

                                 
                # end run over sets 
print("PDF arrays succesfully filled")

#---------------------------------------------------------------------
# Compute central values and uncertainties
#---------------------------------------------------------------------

p1_high = np.nanpercentile(fit1,84,axis=0)
p1_low = np.nanpercentile(fit1,16,axis=0)
p1_mid = ( p1_high + p1_low )/2.
p1_error = ( p1_high - p1_low )/2.

p2_high = np.nanpercentile(fit2,84,axis=0)
p2_low = np.nanpercentile(fit2,16,axis=0)
p2_mid = ( p2_high + p2_low )/2.
p2_error = ( p2_high - p2_low )/2.

p3_high = np.nanpercentile(fit3,84,axis=0)
p3_low = np.nanpercentile(fit3,16,axis=0)
p3_mid = ( p3_high + p3_low )/2.
p3_error = ( p3_high - p3_low )/2.

#---------------------------------------------------------------------
# Plot absolute SFs
#---------------------------------------------------------------------

py.clf()
ncols,nrows=3,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# pdflabels
labelpdf=[r"$F_2^{\nu p}(x,Q,A=1)$",
          r"$F_2^{\bar{\nu} p}(x,Q,A=1)$",
          r"$F_2^{(\nu +\bar{\nu}) p}(x,Q,A=1)$"]

icount=0
for ifl in range(nfl):

    ax = py.subplot(gs[icount])
    p1=ax.plot(X,p1_mid[ifl],ls="solid")
    ax.fill_between(X,p1_high[ifl],p1_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    p3=ax.plot(X,p2_mid[ifl],ls="dashed")
    ax.fill_between(X,p2_high[ifl],p2_low[ifl],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)

    p5=ax.plot(X,p3_mid[ifl],ls="dashdot")
    ax.fill_between(X,p3_high[ifl],p3_low[ifl],color=rescolors[2],alpha=0.2)
    p6=ax.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin,xmax)
  
    if(filelabel=="q100gev"):
        if(ifl==0): ax.set_ylim(0.01,2000)
        if(ifl==1): ax.set_ylim(0.01,2000)
        if(ifl==2): ax.set_ylim(0.01,2000)
            

    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    ax.set_xlabel(r'$x$',fontsize=16)
    
    if(ifl==0):
        ax.text(0.60,0.85,stringQ,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0])],\
                  ["221110-A1-004","NNPDF31sx\_nlo\_as\_0118\_LHCb\_nf\_6","NNPDF40\_nnlo\_as\_01180"], \
                  frameon=True,loc=3,prop={'size':11})
        
    icount = icount + 1

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-'+filelabel+'.pdf')
print('output plot: NNUSF-'+filelabel+'.pdf')

#---------------------------------------------------------------------
# Plot ratios to the proton baseline
#---------------------------------------------------------------------

py.clf()
ncols,nrows=3,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# pdflabels
labelpdf=[r"$F_2^{\nu p}~({\rm ratio~to~ref})$",
          r"$F_2^{\bar{\nu} p}~({\rm ratio~to~ref})$",
          r"$F_2^{(\nu +\bar{\nu}) p}~({\rm ratio~to~ref})$"]
icount=0
for ifl in range(nfl):

    norm = p3_mid[ifl]
    
    ax = py.subplot(gs[icount])
    p1=ax.plot(X,p1_mid[ifl]/norm,ls="solid")
    ax.fill_between(X,p1_high[ifl]/norm,p1_low[ifl]/norm,color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    p3=ax.plot(X,p2_mid[ifl]/norm,ls="dashed")
    ax.fill_between(X,p2_high[ifl]/norm,p2_low[ifl]/norm,color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)

    p5=ax.plot(X,p3_mid[ifl]/norm,ls="dashdot")
    ax.fill_between(X,p3_high[ifl]/norm,p3_low[ifl]/norm,color=rescolors[2],alpha=0.2)
    p6=ax.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlim(xmin,xmax)

    if(filelabel=="q100gev"):
        if(ifl==0): ax.set_ylim(0.5,2.5)
        if(ifl==1): ax.set_ylim(0.5,2.5)
        if(ifl==2): ax.set_ylim(0.5,2.5)
        
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    ax.set_xlabel(r'$x$',fontsize=16)
    
    
    if(ifl==0):
        ax.text(0.15,0.10,stringQ,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0])],\
                  ["221110-A1-004","NNPDF31sx\_nlo\_as\_0118\_LHCb\_nf\_6","NNPDF40\_nnlo\_as\_01180"], \
                  frameon=True,loc=1,prop={'size':11})

    icount = icount + 1

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-ratio-'+filelabel+'.pdf')
print('output plot: NNUSF-ratio-'+filelabel+'.pdf')

exit()




