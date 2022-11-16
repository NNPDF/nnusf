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
logxmin=-4.5
xmax=0.6

############################################################
nset =2 # Number of PDF sets to compare

# PDFs in Lead
pdfsetA=["nNNPDF30_nlo_as_0118_A208_Z82","EPPS21nlo_CT18Anlo_Pb208"]
#pdfsetA=["nNNPDF30_nlo_as_0118_A208_Z82","EPPS16nlo_CT14nlo_Pb208"]

# Baseline PDFs in proton
# Only in NNPDF there is correlation  between protons and heavy nuclei
pdfsetP=["nNNPDF30_nlo_as_0118_p","CT18ANLO"]
#pdfsetP=["nNNPDF30_nlo_as_0118_p","CT14nlo"]

# The PDF set labels
pdfsetlab=[r"${\rm nNNPDF3.0}$", r"${\rm EPPS21}$"]
#pdfsetlab=[r"${\rm nNNPDF3.0}$", r"${\rm EPPS16}$"]
error_option=["mc_90cl","ct"]

q = 10 # GeV
filelabel="-neutrinoSFs_q10gev"

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
    # For EPPS16: run only over lead eigenvectors
    if(iset==1):
        #nrep[iset]=40
        #print("nrep (EPPS16 adjusted) = ", nrep[iset])
        nrep[iset]=48
        print("nrep (EPPS21 adjusted) = ", nrep[iset])
    # For EPPS21: run only over lead eigenvectors
    #if(iset==1):
    #    nrep[iset]=48
    #    print("nrep (EPPS21 adjusted) = ", nrep[iset])
    pP=lhapdf.getPDFSet(pdfsetP[iset])
    print(pP.description)
    
    # Arrays to store LHAPDF results
    if(iset==0):
        fit1 = np.zeros((nrep[iset],nfl,2*nx))
        fit1_cv = np.zeros((nfl,2*nx))
    if(iset==1):
        fit2 = np.zeros((nrep[iset],nfl,2*nx))
        fit2_cv = np.zeros((nfl,2*nx))
    
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
                if(ifl==0): # F2^nu
                    if(iset==0):
                        fit1[i-1][ifl][k] =  ( pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) + pA.xfxQ(-4,x,q) )/\
                            ( pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) + pP.xfxQ(-4,x,q) )
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) + pA.xfxQ(-4,x,q) )/\
                            ( pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) + pP.xfxQ(-4,x,q) )
                #----------------------------------------------------------------
                if(ifl==1): # F2^nubar
                    if(iset==0):
                        fit1[i-1][ifl][k] =  ( pA.xfxQ(+2,x,q) +  pA.xfxQ(-1,x,q) + pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) + pP.xfxQ(-1,x,q) + pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( pA.xfxQ(+2,x,q) +  pA.xfxQ(-1,x,q) + pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) + pP.xfxQ(-1,x,q) + pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                #----------------------------------------------------------------
                if(ifl==2): # F2^nu+
                    if(iset==0):
                        fit1[i-1][ifl][k] = ( pA.xfxQ(+1,x,q)+pA.xfxQ(-1,x,q)+\
                                              pA.xfxQ(+2,x,q)+pA.xfxQ(-2,x,q)+\
                                              pA.xfxQ(+3,x,q)+pA.xfxQ(-3,x,q)+\
                                              pA.xfxQ(+4,x,q)+pA.xfxQ(-4,x,q) ) / \
                                              ( pP.xfxQ(+1,x,q)+pP.xfxQ(-1,x,q)+\
                                                pP.xfxQ(+2,x,q)+pP.xfxQ(-2,x,q)+\
                                                pP.xfxQ(+3,x,q)+pP.xfxQ(-3,x,q)+\
                                                pP.xfxQ(+4,x,q)+pP.xfxQ(-4,x,q) )
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( pA.xfxQ(+1,x,q)+pA.xfxQ(-1,x,q)+\
                                              pA.xfxQ(+2,x,q)+pA.xfxQ(-2,x,q)+\
                                              pA.xfxQ(+3,x,q)+pA.xfxQ(-3,x,q)+\
                                              pA.xfxQ(+4,x,q)+pA.xfxQ(-4,x,q) ) / \
                                              ( pP.xfxQ(+1,x,q)+pP.xfxQ(-1,x,q)+\
                                                pP.xfxQ(+2,x,q)+pP.xfxQ(-2,x,q)+\
                                                pP.xfxQ(+3,x,q)+pP.xfxQ(-3,x,q)+\
                                                pP.xfxQ(+4,x,q)+pP.xfxQ(-4,x,q) )

                #----------------------------------------------------------------
                if(ifl==3): # F3^nu
                    if(iset==0):
                        fit1[i-1][ifl][k] =  ( -pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) - pA.xfxQ(-4,x,q) )/\
                            ( -pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) - pP.xfxQ(-4,x,q) )
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( -pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) - pA.xfxQ(-4,x,q) )/\
                            ( -pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) - pP.xfxQ(-4,x,q) )
                #----------------------------------------------------------------
                if(ifl==4): # F3^nubar
                    if(iset==0):
                        fit1[i-1][ifl][k] =  ( pA.xfxQ(+2,x,q) -  pA.xfxQ(-1,x,q) - pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) - pP.xfxQ(-1,x,q) - pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( pA.xfxQ(+2,x,q) -  pA.xfxQ(-1,x,q) - pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) - pP.xfxQ(-1,x,q) - pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                        
                #----------------------------------------------------------------
                if(ifl==5): # F3^nu+nubar
                    if(iset==0):
                        fit1[i-1][ifl][k] = ( (pA.xfxQ(+1,x,q)-pA.xfxQ(-1,x,q))+\
                                              (pA.xfxQ(+2,x,q)-pA.xfxQ(-2,x,q))+\
                                              (pA.xfxQ(+3,x,q)-pA.xfxQ(-3,x,q)) ) / \
                                              ( (pP.xfxQ(+1,x,q)-pP.xfxQ(-1,x,q))+\
                                                (pP.xfxQ(+2,x,q)-pP.xfxQ(-2,x,q))+\
                                                (pP.xfxQ(+3,x,q)-pP.xfxQ(-3,x,q)) )
                                             
                    if(iset==1):
                        fit2[i-1][ifl][k] = ( (pA.xfxQ(+1,x,q)-pA.xfxQ(-1,x,q))+\
                                              (pA.xfxQ(+2,x,q)-pA.xfxQ(-2,x,q))+\
                                              (pA.xfxQ(+3,x,q)-pA.xfxQ(-3,x,q)) ) / \
                                              ( (pP.xfxQ(+1,x,q)-pP.xfxQ(-1,x,q))+\
                                                (pP.xfxQ(+2,x,q)-pP.xfxQ(-2,x,q))+\
                                                (pP.xfxQ(+3,x,q)-pP.xfxQ(-3,x,q)) )
               
             

# Central values (only for EPPS12)
    pA=lhapdf.mkPDF(pdfsetA[iset],0)
    pP=lhapdf.mkPDF(pdfsetP[iset],0)
    for k in range(2*nx):
        x = X[k]
        
        for ifl in range(nfl):

            #----------------------------------------------------------------
            if(ifl==0): # F2^nu
                if(iset==0):
                    fit1_cv[ifl][k] = ( pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) + pA.xfxQ(-4,x,q) )/\
                            ( pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) + pP.xfxQ(-4,x,q) )
                if(iset==1):
                    fit2_cv[ifl][k] =  ( pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) + pA.xfxQ(-4,x,q) )/\
                            ( pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) + pP.xfxQ(-4,x,q) )
            #----------------------------------------------------------------
            if(ifl==1): # F2^nubar
                if(iset==0):
                    fit1_cv[ifl][k] = ( pA.xfxQ(+2,x,q) +  pA.xfxQ(-1,x,q) + pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) + pP.xfxQ(-1,x,q) + pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                if(iset==1):
                    fit2_cv[ifl][k] = ( pA.xfxQ(+2,x,q) +  pA.xfxQ(-1,x,q) + pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                            ( pP.xfxQ(+2,x,q) + pP.xfxQ(-1,x,q) + pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
            #----------------------------------------------------------------
            if(ifl==2): # F2^nu+
                if(iset==0):
                    fit1_cv[ifl][k] = ( pA.xfxQ(+1,x,q)+pA.xfxQ(-1,x,q)+\
                                              pA.xfxQ(+2,x,q)+pA.xfxQ(-2,x,q)+\
                                              pA.xfxQ(+3,x,q)+pA.xfxQ(-3,x,q)+\
                                              pA.xfxQ(+4,x,q)+pA.xfxQ(-4,x,q) ) / \
                                              ( pP.xfxQ(+1,x,q)+pP.xfxQ(-1,x,q)+\
                                                pP.xfxQ(+2,x,q)+pP.xfxQ(-2,x,q)+\
                                                pP.xfxQ(+3,x,q)+pP.xfxQ(-3,x,q)+\
                                                pP.xfxQ(+4,x,q)+pP.xfxQ(-4,x,q) )
                if(iset==1):
                    fit2_cv[ifl][k] = ( pA.xfxQ(+1,x,q)+pA.xfxQ(-1,x,q)+\
                                              pA.xfxQ(+2,x,q)+pA.xfxQ(-2,x,q)+\
                                              pA.xfxQ(+3,x,q)+pA.xfxQ(-3,x,q)+\
                                              pA.xfxQ(+4,x,q)+pA.xfxQ(-4,x,q) ) / \
                                              ( pP.xfxQ(+1,x,q)+pP.xfxQ(-1,x,q)+\
                                                pP.xfxQ(+2,x,q)+pP.xfxQ(-2,x,q)+\
                                                pP.xfxQ(+3,x,q)+pP.xfxQ(-3,x,q)+\
                                                pP.xfxQ(+4,x,q)+pP.xfxQ(-4,x,q) )
            #----------------------------------------------------------------
            if(ifl==3): # F3^nu
                if(iset==0):
                    fit1_cv[ifl][k]  =  ( -pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) - pA.xfxQ(-4,x,q) )/\
                            ( -pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) - pP.xfxQ(-4,x,q) )
                if(iset==1):
                    fit2_cv[ifl][k]  = ( -pA.xfxQ(-2,x,q) +  pA.xfxQ(+1,x,q) + pA.xfxQ(+3,x,q) - pA.xfxQ(-4,x,q) )/\
                            ( -pP.xfxQ(-2,x,q) + pP.xfxQ(+1,x,q) + pP.xfxQ(+3,x,q) - pP.xfxQ(-4,x,q) )
            #----------------------------------------------------------------
            if(ifl==4): # F3^nubar
                if(iset==0):
                    fit1_cv[ifl][k]  =  ( pA.xfxQ(+2,x,q) -  pA.xfxQ(-1,x,q) - pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                        ( pP.xfxQ(+2,x,q) - pP.xfxQ(-1,x,q) - pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                if(iset==1):
                    fit2_cv[ifl][k]  = ( pA.xfxQ(+2,x,q) -  pA.xfxQ(-1,x,q) - pA.xfxQ(-3,x,q) + pA.xfxQ(+4,x,q) )/\
                        ( pP.xfxQ(+2,x,q) - pP.xfxQ(-1,x,q) - pP.xfxQ(-3,x,q) + pP.xfxQ(+4,x,q) )
                        
            #----------------------------------------------------------------
            if(ifl==5): # F3^nu+nubar
                if(iset==0):
                    fit1_cv[ifl][k]  = ( (pA.xfxQ(+1,x,q)-pA.xfxQ(-1,x,q))+\
                                          (pA.xfxQ(+2,x,q)-pA.xfxQ(-2,x,q))+\
                                          (pA.xfxQ(+3,x,q)-pA.xfxQ(-3,x,q)) ) / \
                                          ( (pP.xfxQ(+1,x,q)-pP.xfxQ(-1,x,q))+\
                                            (pP.xfxQ(+2,x,q)-pP.xfxQ(-2,x,q))+\
                                            (pP.xfxQ(+3,x,q)-pP.xfxQ(-3,x,q)) )
                                             
                if(iset==1):
                    fit2_cv[ifl][k]  = ( (pA.xfxQ(+1,x,q)-pA.xfxQ(-1,x,q))+\
                                          (pA.xfxQ(+2,x,q)-pA.xfxQ(-2,x,q))+\
                                          (pA.xfxQ(+3,x,q)-pA.xfxQ(-3,x,q)) ) / \
                                          ( (pP.xfxQ(+1,x,q)-pP.xfxQ(-1,x,q))+\
                                            (pP.xfxQ(+2,x,q)-pP.xfxQ(-2,x,q))+\
                                            (pP.xfxQ(+3,x,q)-pP.xfxQ(-3,x,q)) )
           
        

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
            
    # CT: asymmetric Hessian with then normalisation to one-sigma
    elif(error_option[iset]=="ct"):

        if(iset==1):
            p2_mid = np.mean(fit2,axis=0)
            p2_error = np.std(fit2,axis=0)
            neig = int(nrep[iset]/2) # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(2*nx):
                for ifl in range(nfl):
                    p2_mid[ifl][ix]= fit2_cv[ifl][ix]
                    p2_error[ifl][ix]=0 # initialisation
                    for ieg in range(neig):
                        #print(2*ieg+1,nrep[iset])
                        p2_error[ifl][ix] = p2_error[ifl][ix] \
                            + (fit2[2*ieg+1][ifl][ix] - fit2[2*ieg][ifl][ix] )**2.0
                    p2_error[ifl][ix]=math.sqrt(p2_error[ifl][ix])/2
            p2_high = p2_mid + p2_error 
            p2_low = p2_mid - p2_error
       
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
labelpdf=[r"$R^{\rm Pb}_{F_2^\nu}(x,Q)$",r"$R^{\rm Pb}_{F_2^{\bar{\nu}}}(x,Q)$",\
          r"$R^{\rm Pb}_{F_2^{\nu+\bar{\nu}}}(x,Q)$",\
          r"$R^{\rm Pb}_{F_3^\nu}(x,Q)$",r"$R^{\rm Pb}_{F_3^{\bar{\nu}}}(x,Q)$",\
          r"$R^{\rm Pb}_{F_3^{\nu+\bar{\nu}}}(x,Q)$"]
yranges=[[0.5,1.50],[0.5,1.50],[0.5,1.50],[0.2,2.2],[0.2,2.2],[0.2,2.2]]

for ifl in range(nfl):

    ax = py.subplot(gs[ifl])
          
    p1=ax.plot(X,p1_mid[ifl],ls="dashed")
    ax.fill_between(X,p1_high[ifl],p1_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    p3=ax.plot(X,p2_mid[ifl],ls="solid")
    ax.fill_between(X,p2_high[ifl],p2_low[ifl],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
       
    ax.set_xscale('log')
    ax.set_xlim(10**(logxmin),xmax)
    if(ifl>2):
        ax.set_xlim(0.009,xmax)
    ax.tick_params(which='both',direction='in',labelsize=15,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=21)
    ax.set_ylim(yranges[ifl][0],yranges[ifl][1])
    if(ifl>2):
        ax.set_xlabel(r'$x$',fontsize=17)
    ax.axhline(1, color='black', linewidth=0.8,ls="dashed")

    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0])],[pdfsetlab[0],pdfsetlab[1]],\
                  frameon=True,loc="best",prop={'size':15})
    
    if(ifl==1):
        string=r'$Q='+str(q)+'~{\\rm GeV}$'
        ax.text(0.40,0.85,string,fontsize=20,transform=ax.transAxes)
        
py.tight_layout(pad=1, w_pad=1.3, h_pad=1.0)
py.savefig('nuclearcorr'+filelabel+'.pdf')
print('output plot: nuclearcorr'+filelabel+'.pdf')
    

#*******************************************************************************************************


