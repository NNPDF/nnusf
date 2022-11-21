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
xmin = 1e-8
xmax= 9e-1
# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=1000
# number of flavours to be plotted
nfl=9
# Set x grid
X = np.logspace(log(xmin),log(xmax),nx)

# number of pdf sets
nset=3

nrep=np.zeros(nset, dtype='int')
nrep_max = 100

pdfset=["221110-A1-004","221110-A56-004","221110-A208-004"]
fit1 = np.zeros((nrep_max,nfl,nx))
fit2 = np.zeros((nrep_max,nfl,nx))
fit3 = np.zeros((nrep_max,nfl,nx))

ids = np.array([1001,1002,1003,2001,2002,2003,3001,3002,3003])

for iset in range(nset):

    p=lhapdf.getPDFSet(pdfset[iset])
    nrep[iset]=int(p.get_entry("NumMembers"))-1
    print("nrep = ",nrep[iset])
    if(nrep[iset] > nrep_max):
        nrep[iset] = nrep_max
    if(nrep[iset] > nrep_max):
        print("Problem, too many replicas \n")
        exit()
    #print(p.description)
    print("nrep (fixed) = ",nrep[iset])

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

                #print(x," ",q)
                if(iset==0):
                    fit1[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==1):
                    fit2[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==2):
                    fit3[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                #print(i," ",ifl," ",fit1[i-1][ifl][k])
                
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
ncols,nrows=3,3
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# pdflabels
labelpdf=[r"$F_2^{\nu p}(x,Q,A)$",
          r"$F_L^{\nu p}(x,Q,A)$",
          r"$xF_3^{\nu p}(x,Q,A)$",\
          r"$F_2^{\bar{\nu} p}(x,Q,A)$",
          r"$F_L^{\bar{\nu} p}(x,Q,A)$",
          r"$xF_3^{\bar{\nu} p}(x,Q,A)$",\
          r"$F_2^{(\nu +\bar{\nu}) p}(x,Q,A)$",\
          r"$F_L^{(\nu +\bar{\nu}) p}(x,Q,A)$",\
          r"$xF_3^{(\nu +\bar{\nu}) p}(x,Q,A)$",]

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
    ax.set_xlim(xmin,xmax)

    if(filelabel=="q0p5gev"):
        if(ifl==0): ax.set_ylim(0,1.5)
        if(ifl==3): ax.set_ylim(0,1.5)
        if(ifl==6): ax.set_ylim(0,1.5) 
        if(ifl==1): ax.set_ylim(-0.1,0.75)
        if(ifl==4): ax.set_ylim(-0.1,0.75)
        if(ifl==7): ax.set_ylim(-0.1,0.75)
        if(ifl==2): ax.set_ylim(-0.3,0.55)
        if(ifl==5): ax.set_ylim(-0.3,0.55)
        if(ifl==8): ax.set_ylim(-0.3,0.55)

    if(filelabel=="q1gev"):
        if(ifl==0): ax.set_ylim(0,4)
        if(ifl==3): ax.set_ylim(0,4)
        if(ifl==6): ax.set_ylim(0,4) 
        if(ifl==1): ax.set_ylim(-0.5,2.5)
        if(ifl==4): ax.set_ylim(-0.5,2.5)
        if(ifl==7): ax.set_ylim(-0.5,2.5)
        if(ifl==2): ax.set_ylim(-0.6,1.4)
        if(ifl==5): ax.set_ylim(-0.6,1.4)
        if(ifl==8): ax.set_ylim(-0.6,1.4)

    if(filelabel=="q5p0gev"):
        if(ifl==0): ax.set_ylim(0,5)
        if(ifl==3): ax.set_ylim(0,5)
        if(ifl==6): ax.set_ylim(0,5) 
        if(ifl==1): ax.set_ylim(-0.5,3.0)
        if(ifl==4): ax.set_ylim(-0.5,3.0)
        if(ifl==7): ax.set_ylim(-0.5,3.0)
        if(ifl==2): ax.set_ylim(-0.6,1.4)
        if(ifl==5): ax.set_ylim(-0.6,1.4)
        if(ifl==8): ax.set_ylim(-0.6,1.4)

    if(filelabel=="q14p0gev"):
        if(ifl==0): ax.set_ylim(0,5)
        if(ifl==3): ax.set_ylim(0,5)
        if(ifl==6): ax.set_ylim(0,5) 
        if(ifl==1): ax.set_ylim(-0.5,3.0)
        if(ifl==4): ax.set_ylim(-0.5,3.0)
        if(ifl==7): ax.set_ylim(-0.5,3.0)
        if(ifl==2): ax.set_ylim(-0.6,1.4)
        if(ifl==5): ax.set_ylim(-0.6,1.4)
        if(ifl==8): ax.set_ylim(-0.6,1.4)

    if(filelabel=="q1p58gev"):
        if(ifl==0): ax.set_ylim(0,5)
        if(ifl==3): ax.set_ylim(0,5)
        if(ifl==6): ax.set_ylim(0,5) 
        if(ifl==1): ax.set_ylim(-0.5,3.0)
        if(ifl==4): ax.set_ylim(-0.5,3.0)
        if(ifl==7): ax.set_ylim(-0.5,3.0)
        if(ifl==2): ax.set_ylim(-0.6,1.4)
        if(ifl==5): ax.set_ylim(-0.6,1.4)
        if(ifl==8): ax.set_ylim(-0.6,1.4)

    if(filelabel=="q100gev"):
        if(ifl==0): ax.set_ylim(0.01,1000)
        if(ifl==3): ax.set_ylim(0.01,1000)
        if(ifl==6): ax.set_ylim(0.01,1000) 
        if(ifl==1): ax.set_ylim(0.01,1000)
        if(ifl==4): ax.set_ylim(0.01,1000)
        if(ifl==7): ax.set_ylim(0.01,1000)
        if(ifl==0 or ifl==3 or ifl==6 or ifl==1 or ifl==4 or ifl==7):
            ax.set_yscale('log')
        if(ifl==2): ax.set_ylim(0,1.8)
        if(ifl==5): ax.set_ylim(-1.2,1.4)
        if(ifl==8): ax.set_ylim(-0.2,1.0)
    

    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$x$',fontsize=16)
    
    if(ifl==0):
        ax.text(0.60,0.85,stringQ,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==3):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0])],\
                  [ r"$A=1$",r"$A=56$", r"$A=208$" ], \
                  frameon=True,loc=3,prop={'size':15})
        
    icount = icount + 1

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-'+filelabel+'.pdf')
print('output plot: NNUSF-'+filelabel+'.pdf')

#---------------------------------------------------------------------
# Plot ratios to the proton baseline
#---------------------------------------------------------------------

py.clf()
ncols,nrows=3,3
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# pdflabels
labelpdf=[r"$F_2^{\nu p}~({\rm ratio~to~}A=1)$",
          r"$F_L^{\nu p}~({\rm ratio~to~}A=1)$",
          r"$xF_3^{\nu p}~({\rm ratio~to~}A=1)$",\
          r"$F_2^{\bar{\nu} p}~({\rm ratio~to~}A=1)$",
          r"$F_L^{\bar{\nu} p}~({\rm ratio~to~}A=1)$",
          r"$xF_3^{\bar{\nu} p}~({\rm ratio~to~}A=1)$",\
          r"$F_2^{(\nu +\bar{\nu}) p}~({\rm ratio~to~}A=1)$",\
          r"$F_L^{(\nu +\bar{\nu}) p}~({\rm ratio~to~}A=1)$",\
          r"$xF_3^{(\nu +\bar{\nu}) p}~({\rm ratio~to~}A=1)$",]

icount=0
for ifl in range(nfl):

    norm = p1_mid[ifl]
    
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

    if(filelabel=="q0p5gev"):
        if(ifl==0): ax.set_ylim(0,2)
        if(ifl==3): ax.set_ylim(0,2)
        if(ifl==6): ax.set_ylim(0,2) 
        if(ifl==1): ax.set_ylim(0,2)
        if(ifl==4): ax.set_ylim(0,2)
        if(ifl==7): ax.set_ylim(0,2)
        if(ifl==2): ax.set_ylim(0,2)
        if(ifl==5): ax.set_ylim(0,2)
        if(ifl==8): ax.set_ylim(0,2)

    if(filelabel=="q1gev"):
        if(ifl==0): ax.set_ylim(0,2)
        if(ifl==3): ax.set_ylim(0,2)
        if(ifl==6): ax.set_ylim(0,2) 
        if(ifl==1): ax.set_ylim(0,2)
        if(ifl==4): ax.set_ylim(0,2)
        if(ifl==7): ax.set_ylim(0,2)
        if(ifl==2): ax.set_ylim(0,2)
        if(ifl==5): ax.set_ylim(0,2)
        if(ifl==8): ax.set_ylim(0,2)      

    if(filelabel=="q5p0gev"):
        if(ifl==0): ax.set_ylim(0,2)
        if(ifl==3): ax.set_ylim(0,2)
        if(ifl==6): ax.set_ylim(0,2) 
        if(ifl==1): ax.set_ylim(0,3)
        if(ifl==4): ax.set_ylim(0,3)
        if(ifl==7): ax.set_ylim(0,3)
        if(ifl==2): ax.set_ylim(0,2)
        if(ifl==5): ax.set_ylim(0,2)
        if(ifl==8): ax.set_ylim(0,2)

    if(filelabel=="q14p0gev"):
        if(ifl==0): ax.set_ylim(0,2)
        if(ifl==3): ax.set_ylim(0,2)
        if(ifl==6): ax.set_ylim(0,2) 
        if(ifl==1): ax.set_ylim(0,3)
        if(ifl==4): ax.set_ylim(0,3)
        if(ifl==7): ax.set_ylim(0,3)
        if(ifl==2): ax.set_ylim(0,2)
        if(ifl==5): ax.set_ylim(0,2)
        if(ifl==8): ax.set_ylim(0,2)

    if(filelabel=="q100gev"):
        if(ifl==0): ax.set_ylim(0,2)
        if(ifl==3): ax.set_ylim(0,2)
        if(ifl==6): ax.set_ylim(0,2) 
        if(ifl==1): ax.set_ylim(0,3)
        if(ifl==4): ax.set_ylim(0,3)
        if(ifl==7): ax.set_ylim(0,3)
        if(ifl==2): ax.set_ylim(0,2)
        if(ifl==5): ax.set_ylim(0,2)
        if(ifl==8): ax.set_ylim(0,2)  

    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$x$',fontsize=16)
    
    
    if(ifl==0):
        ax.text(0.60,0.85,stringQ,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==3):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0])],\
                  [ r"$A=1$", r"$A=56$", r"$A=208$" ], \
                  frameon=True,loc=1,prop={'size':16})

    icount = icount + 1

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-ratio-'+filelabel+'.pdf')
print('output plot: NNUSF-ratio-'+filelabel+'.pdf')

exit()




