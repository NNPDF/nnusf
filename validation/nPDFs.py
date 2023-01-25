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

#---------------------------------------------------------
#---------------------------------------------------------
# General plot settings
nx = 400
logxmin=-6
xmax=0.8

############################################################
nset =4 # Number of PDF sets to compare

# PDFs in Lead
pdfsetA=["nNNPDF30_nlo_as_0118_A56_Z26","nNNPDF30_nlo_as_0118_p_Fe56",\
         "nNNPDF30_nlo_as_0118_A208_Z82","nNNPDF30_nlo_as_0118_p_Pb208"]

# Baseline PDFs in proton
pdfsetP=["nNNPDF30_nlo_as_0118_p","nNNPDF30_nlo_as_0118_p","nNNPDF30_nlo_as_0118_p","nNNPDF30_nlo_as_0118_p"]

# The PDF set labels
pdfsetlab=[r"$A=56, Z=26~({\rm bound~nucleon})$",\
           r"$A=56~({\rm bound~proton})$",\
           r"$A=208, Z=82~({\rm bound~nucleon})$",\
           r"$A=208~({\rm bound~proton})$"]

q = 10 # GeV
error_option=["mc_90cl","mc_90cl","mc_90cl","mc_90cl"]
filelabel="-neutrinoPDFs_q10gev"

# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=2001
# number of flavours to be plotted
nfl=6
# Set x grid
X = np.concatenate([np.logspace(logxmin,-1,nx),np.linspace(0.11,xmax,nx)],axis=0)

# Threshold for ratios
eps=1e-10

# Number of replicas
nrep=np.zeros(nset, dtype=int)

print("\n Filling PDF arrays from LHAPDF \n")

# run over PDF sets
for iset in range(nset):
    
    # Initialise PDF set
    print("\nPDF set grid = ",pdfsetA[iset])
    pA=lhapdf.getPDFSet(pdfsetA[iset])
    print(pA.description)
    nrep[iset] = int(pA.get_entry("NumMembers"))-1
    print("nrep =", nrep[iset])
    print("PDF set grid = ",pdfsetP[iset])
    pP=lhapdf.getPDFSet(pdfsetP[iset])
    print(pP.description)
    
    # Arrays to store LHAPDF results
    if(iset==0):
        fit1 = np.zeros((nrep[iset],nfl,2*nx))
        fit1_cv = np.zeros((nfl,2*nx))
    if(iset==1):
        fit2 = np.zeros((nrep[iset],nfl,2*nx))
        fit2_cv = np.zeros((nfl,2*nx))
    if(iset==2):
        fit3 = np.zeros((nrep[iset],nfl,2*nx))
        fit3_cv = np.zeros((nfl,2*nx))
    if(iset==3):
        fit4 = np.zeros((nrep[iset],nfl,2*nx))
        fit4_cv = np.zeros((nfl,2*nx))
    
    # Run over replicas
    for i in range(1,nrep[iset]+1):
        # The correct replica/eigenvector
        pA=lhapdf.mkPDF(pdfsetA[iset],i)
        if(iset==0):
            pP=lhapdf.mkPDF(pdfsetP[iset],i) # The corresponding proton replica for nNNPDF
        if(iset>0):
            pP=lhapdf.mkPDF(pdfsetP[iset],0) # Central value for EPPS21 
                
        # Run over x arrat
        for k in range(2*nx):
            
            x = X[k]

            # run over flavours
            for ifl in range(nfl):

                #----------------------------------------------------------------
                if(ifl==0): # u
                    if(iset==0):
                        fit1[i-1][ifl][k] =  pA.xfxQ(+2,x,q)/pP.xfxQ(+2,x,q)                                          
                    if(iset==1):
                        fit2[i-1][ifl][k] = pA.xfxQ(+2,x,q)/pP.xfxQ(+2,x,q)
                    if(iset==2):
                        fit3[i-1][ifl][k] =  pA.xfxQ(+2,x,q)/pP.xfxQ(+2,x,q)                                          
                    if(iset==3):
                        fit4[i-1][ifl][k] = pA.xfxQ(+2,x,q)/pP.xfxQ(+2,x,q)
                #----------------------------------------------------------------
                if(ifl==1): # d
                    if(iset==0):
                        fit1[i-1][ifl][k] =  pA.xfxQ(+1,x,q)/pP.xfxQ(+1,x,q)                                             
                    if(iset==1):
                        fit2[i-1][ifl][k] = pA.xfxQ(+1,x,q)/pP.xfxQ(+1,x,q)
                    if(iset==2):
                        fit3[i-1][ifl][k] =  pA.xfxQ(+1,x,q)/pP.xfxQ(+1,x,q)                                             
                    if(iset==3):
                        fit4[i-1][ifl][k] = pA.xfxQ(+1,x,q)/pP.xfxQ(+1,x,q)
                #----------------------------------------------------------------
                if(ifl==2): # sigma
                    if(iset==0):
                        fit1[i-1][ifl][k] =  (pA.xfxQ(+1,x,q)+pA.xfxQ(+2,x,q)+pA.xfxQ(-1,x,q)+pA.xfxQ(-2,x,q))/\
                            (pP.xfxQ(+1,x,q)+pP.xfxQ(+2,x,q)  +pP.xfxQ(-1,x,q)+pP.xfxQ(-2,x,q)    )                           
                    if(iset==1):
                        fit2[i-1][ifl][k] =  (pA.xfxQ(+1,x,q)+pA.xfxQ(+2,x,q)+pA.xfxQ(-1,x,q)+pA.xfxQ(-2,x,q))/\
                            (pP.xfxQ(+1,x,q)+pP.xfxQ(+2,x,q)  +pP.xfxQ(-1,x,q)+pP.xfxQ(-2,x,q)    ) 
                    if(iset==2):
                        fit3[i-1][ifl][k] =  (pA.xfxQ(+1,x,q)+pA.xfxQ(+2,x,q)+pA.xfxQ(-1,x,q)+pA.xfxQ(-2,x,q))/\
                            (pP.xfxQ(+1,x,q)+pP.xfxQ(+2,x,q)  +pP.xfxQ(-1,x,q)+pP.xfxQ(-2,x,q)    ) 
                    if(iset==3):
                        fit4[i-1][ifl][k] =  (pA.xfxQ(+1,x,q)+pA.xfxQ(+2,x,q)+pA.xfxQ(-1,x,q)+pA.xfxQ(-2,x,q))/\
                            (pP.xfxQ(+1,x,q)+pP.xfxQ(+2,x,q)  +pP.xfxQ(-1,x,q)+pP.xfxQ(-2,x,q)    ) 
                #----------------------------------------------------------------
                if(ifl==3): # ubar
                    if(iset==0):
                        fit1[i-1][ifl][k] =  pA.xfxQ(-2,x,q)/pP.xfxQ(-2,x,q)                                          
                    if(iset==1):
                        fit2[i-1][ifl][k] = pA.xfxQ(-2,x,q)/pP.xfxQ(-2,x,q)
                    if(iset==2):
                        fit3[i-1][ifl][k] =  pA.xfxQ(-2,x,q)/pP.xfxQ(-2,x,q)                                          
                    if(iset==3):
                        fit4[i-1][ifl][k] = pA.xfxQ(-2,x,q)/pP.xfxQ(-2,x,q)
                #----------------------------------------------------------------
                if(ifl==4): # dbar
                    if(iset==0):
                        fit1[i-1][ifl][k] =  pA.xfxQ(-1,x,q)/pP.xfxQ(-1,x,q)                                             
                    if(iset==1):
                        fit2[i-1][ifl][k] = pA.xfxQ(-1,x,q)/pP.xfxQ(-1,x,q)
                    if(iset==2):
                        fit3[i-1][ifl][k] =  pA.xfxQ(-1,x,q)/pP.xfxQ(-1,x,q)                                             
                    if(iset==3):
                        fit4[i-1][ifl][k] = pA.xfxQ(-1,x,q)/pP.xfxQ(-1,x,q)
                #----------------------------------------------------------------
                if(ifl==5): # gluon
                    if(iset==0):
                        fit1[i-1][ifl][k] =  pA.xfxQ(0,x,q)/pP.xfxQ(0,x,q)                                             
                    if(iset==1):
                        fit2[i-1][ifl][k] = pA.xfxQ(0,x,q)/pP.xfxQ(0,x,q)
                    if(iset==2):
                        fit3[i-1][ifl][k] =  pA.xfxQ(0,x,q)/pP.xfxQ(0,x,q)                                             
                    if(iset==3):
                        fit4[i-1][ifl][k] = pA.xfxQ(0,x,q)/pP.xfxQ(0,x,q)

#---------------------------------------------------------------------
# Compute central values and uncertainties
#---------------------------------------------------------------------

for iset in range(nset):

    if(error_option[iset]=="mc_90cl"):

        if(iset==0):
            p1_high = np.nanpercentile(fit1,95,axis=0)
            p1_low = np.nanpercentile(fit1,5,axis=0)
            p1_mid = ( p1_high + p1_low )/2.
            p1_mid = np.median(fit1,axis=0)
            p1_error = ( p1_high - p1_low )/2.

        elif(iset==1):
            p2_high = np.nanpercentile(fit2,95,axis=0)
            p2_low = np.nanpercentile(fit2,5,axis=0)
            p2_mid = ( p2_high + p2_low )/2.
            p2_mid = np.median(fit2,axis=0)
            p2_error = ( p2_high - p2_low )/2.

        elif(iset==2):
            p3_high = np.nanpercentile(fit3,95,axis=0)
            p3_low = np.nanpercentile(fit3,5,axis=0)
            p3_mid = ( p3_high + p3_low )/2.
            p3_mid = np.median(fit3,axis=0)
            p3_error = ( p3_high - p3_low )/2.

        elif(iset==3):
            p4_high = np.nanpercentile(fit4,95,axis=0)
            p4_low = np.nanpercentile(fit4,5,axis=0)
            p4_mid = ( p4_high + p4_low )/2.
            p4_mid = np.median(fit4,axis=0)
            p4_error = ( p4_high - p4_low )/2.
       
    else:
        print("Invalid error option = ",error_option[iset])
        exit()
    
#----------------------------------------------------------------------

#*****************************************************************************
#*****************************************************************************

#---------------------------------------------------------------------
# Plot PDFs both in absolute scale and as ratios to some reference
#---------------------------------------------------------------------

ncols,nrows=3,2
#ncols, nrows = 2,3
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

# First the absolute plots

# pdflabels
labelpdf=[r"$R^{\rm (A)}_{u}(x,Q)$",r"$R^{\rm (A)}_{d}(x,Q)$",r"$R^{\rm (A)}_{\Sigma}(x,Q)$",\
          r"$R^{\rm (A)}_{\bar{u}}(x,Q)$",r"$R^{\rm (A)}_{\bar{d}}(x,Q)$",r"$R^{\rm (A)}_{\bar{g}}(x,Q)$"]
yranges=[[0.41,1.80],[0.41,1.80],[0.41,1.80],[0.41,1.80],[0.41,1.80],[0.41,1.80]]

for ifl in range(nfl):

    ax = py.subplot(gs[ifl])
          
    p1=ax.plot(X,p1_mid[ifl],ls="dashed",color=rescolors[0])
    ax.fill_between(X,p1_high[ifl],p1_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    p3=ax.plot(X,p2_mid[ifl],ls="solid",color=rescolors[2])
    ax.fill_between(X,p2_high[ifl],p2_low[ifl],color=rescolors[2],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)

    p5=ax.plot(X,p3_mid[ifl],ls="dashdot",color=rescolors[1])
    ax.fill_between(X,p3_high[ifl],p3_low[ifl],color=rescolors[1],alpha=0.2)
    p6=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
    p7=ax.plot(X,p4_mid[ifl],ls="dotted",color=rescolors[3])
    ax.fill_between(X,p4_high[ifl],p4_low[ifl],color=rescolors[3],alpha=0.2)
    p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
       
    ax.set_xscale('log')
    ax.set_xlim(10**(logxmin),xmax)
    if(ifl>2):
        ax.set_xlim(10**(logxmin),0.3)
    ax.tick_params(which='both',direction='in',labelsize=15,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=21)
    ax.set_ylim(yranges[ifl][0],yranges[ifl][1])
    if(ifl>2):
        ax.set_xlabel(r'$x$',fontsize=18)
    ax.axhline(1, color='black', linewidth=0.8,ls="dashed")

    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0]),(p7[0],p8[0])],\
                  [pdfsetlab[0],pdfsetlab[1],pdfsetlab[2],pdfsetlab[3]],\
                  frameon=True,loc=2,prop={'size':11})
    
    if(ifl==1):
        string=r'$Q='+str(q)+'~{\\rm GeV}$'
        ax.text(0.20,0.85,string,fontsize=18,transform=ax.transAxes)
        
py.tight_layout(pad=1, w_pad=1.3, h_pad=1.0)
py.savefig('nuclearcorr'+filelabel+'.pdf')
print('output plot: nuclearcorr'+filelabel+'.pdf')
    

#*******************************************************************************************************


