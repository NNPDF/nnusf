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

#---------------------------------------------------------
#---------------------------------------------------------
# General plot settings
q = 1 # GeV
filelabel = "q1gev"
nx = 200
xmin = 1e-4
xmax= 9e-1
# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=1000
# number of flavours to be plotted
nfl=9
# Set x grid
X = np.logspace(log(xmin),log(xmax),nx)

# Number of replicas
pdfset="nnusf_matched"
p=lhapdf.getPDFSet(pdfset)
nrep=int(p.get_entry("NumMembers"))-1
print("nrep = ",nrep)
print(p.description)

fit1 = np.zeros((nrep,nfl,nx))

ids = np.array([1001,1002,1003,2001,2002,2003,3001,3002,3003])
    
# Run over replicas
for i in range(1,nrep+1):
    p=lhapdf.mkPDF(pdfset,i)
    lhapdf.setVerbosity(0)

    # Run over x arrat
    for k in range(nx):
            
        x = X[k]
        q2 = pow(q,2.0)

        # run over flavours
        for ifl in range(nfl):

            #print(x," ",q)
            fit1[i-1][ifl][k] =  p.xfxQ(ids[ifl],x,q)
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

#---------------------------------------------------------------------
# Plot SFs
#---------------------------------------------------------------------


py.clf()
ncols,nrows=3,3
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# First the absolute plots

# pdflabels
labelpdf=[r"$F_2^{\nu p}(x,Q,A)$",r"$xF_3^{\nu p}(x,Q,A)$",r"$F_L^{\nu p}(x,Q,A)$",\
          r"$F_2^{\bar{\nu} p}(x,Q,A)$",r"$xF_3^{\bar{\nu} p}(x,Q,A)$",r"$F_L^{\bar{\nu} p}(x,Q,A)$",\
          r"$F_2^{(\nu +\bar{\nu}) p}(x,Q,A)$",\
          r"$xF_3^{(\nu +\bar{\nu}) p}(x,Q,A)$",\
          r"$F_L^{(\nu +\bar{\nu}) p}(x,Q,A)$"]

icount=0
for ifl in range(nfl):

    ax = py.subplot(gs[icount])
    p1=ax.plot(X,p1_mid[ifl],ls="solid")
    ax.fill_between(X,p1_high[ifl],p1_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlim(xmin,xmax)
 
    if(ifl==0): ax.set_ylim(0,4)
    if(ifl==3): ax.set_ylim(0,4)
    if(ifl==6): ax.set_ylim(0,4) 

    if(ifl==1): ax.set_ylim(0,3)
    if(ifl==4): ax.set_ylim(0,3)
    if(ifl==7): ax.set_ylim(0,3)

    if(ifl==2): ax.set_ylim(-1,1.5)
    if(ifl==5): ax.set_ylim(-1,1.5)
    if(ifl==8): ax.set_ylim(-1,1.5)
    
    
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$x$',fontsize=16)
    
    string=r'$Q=1~{\rm GeV},~A=1$'
    if(ifl==0):
        ax.text(0.50,0.80,string,fontsize=17,transform=ax.transAxes)

    icount = icount + 1

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-'+filelabel+'.pdf')
print('output plot: NNUSF-'+filelabel+'.pdf')



exit()




