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
nq = 200
qmin = 1.65
qmax=6
# set x grid
Q = np.logspace(np.log10(qmin),np.log10(qmax),nq)

print(Q)

# Number of PDF sets to be compared
nset =3
# ----------------------------------------------------------------
# modernPDFs
#pdfset=["CT18NNLO","NNPDF40_nnlo_as_01180","MSHT20nnlo_as118"]
#pdfsetlab=[r"${\rm CT18}$",r"${\rm NNPDF4.0}$",r"${\rm MSHT20}$"]
# error_option=["ct","mc_68cl","mmht"]

# oldPDFs
pdfset=["JR14NLO08VF","NNPDF40_nnlo_as_01180","GRV98lo_patched"]
pdfsetlab=[r"${\rm LO~SF+JR14NLO}$",r"${\rm LO~SF+NNPDF4.0NNLO}$",r"${\rm LO~SF+GRV98LO}$"]
error_option=["symmhessian","mc_68cl","ct"]
filelabel="-oldpdfs_x0p25"
#filelabel="-oldpdfs_x0p0126"


#----------------------------------------------
#----------------------------------------------
# Value of x
x = 0.25
#x = 0.0126
#----------------------------------------------
#----------------------------------------------

#---------------------------
#---------------------------
print("\n Reading the GENIE structure functions \n")

# Read genie inputs
#genie_sf=np.loadtxt("Genie_Data/Genie-F2-xF3-BodekYang_x_0p0126.txt")
genie_sf=np.loadtxt("Genie_Data/Genie-F2-xF3-BodekYang_x0p25.txt")


print(genie_sf)

nq2_genie=111
genie_sf_q=np.zeros(nq2_genie)
genie_sf_f2=np.zeros(nq2_genie)
genie_sf_f3=np.zeros(nq2_genie)

for iq2 in range(nq2_genie):
    genie_sf_q[iq2] = math.sqrt( genie_sf[iq2][1] )
    genie_sf_f2[iq2] = genie_sf[iq2][2]
    genie_sf_f3[iq2] = x* genie_sf[iq2][3] # Genie gives F3 instead of xF3

Q = genie_sf_q     
nq = Q.size

print(genie_sf_q)    
print(genie_sf_f2)
print(genie_sf_f3)

#---------------------------
#---------------------------
print("\n Reading the YADISM structure functions \n")

# Read YADISM inputs
#yadism_lo_sf_f2=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F2_rep0_x0p0126.txt")
#yadism_nnlo_sf_f2=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F2_rep0_x0p0126.txt")
#yadism_lo_sf_f3=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F3_rep0_x0p0126.txt")
#yadism_nnlo_sf_f3=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F3_rep0_x0p0126.txt")

yadism_lo_sf_f2=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F2_rep0_x0p25.txt")
yadism_nnlo_sf_f2=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F2_rep0_x0p25.txt")
yadism_lo_sf_f3=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F3_rep0_x0p25.txt")
yadism_nnlo_sf_f3=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F3_rep0_x0p25.txt")

nq2_yadism=20
yadism_sf_q=np.zeros(nq2_yadism)
yadism_f2=np.zeros(nq2_yadism)
yadism_f3=np.zeros(nq2_yadism)
yadism_nnlo_f2=np.zeros(nq2_yadism)
yadism_nnlo_f3=np.zeros(nq2_yadism)

for iq2 in range(nq2_yadism):
    yadism_sf_q[iq2] = math.sqrt( yadism_lo_sf_f2[iq2][2] )
    yadism_f2[iq2] = yadism_lo_sf_f2[iq2][3]
    yadism_f3[iq2] = yadism_lo_sf_f3[iq2][3]
    yadism_nnlo_f2[iq2] = yadism_nnlo_sf_f2[iq2][3]
    yadism_nnlo_f3[iq2] = yadism_nnlo_sf_f3[iq2][3]
   

#print(yadism_sf_q)    
#print(yadism_sf_f2)
print(yadism_nnlo_f3)
print(yadism_f3)
#exit()


#---------------------------
#---------------------------


# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=100
# Number of replicas
nrep=np.zeros(nset, dtype=int)
# number of Structure Functions to be plotted
nsf=4
# Set x grid

# run over PDF sets
for iset in range(nset):
    # Initialise PDF set
    p=lhapdf.getPDFSet(pdfset[iset])
    print(p.description)
    nrep[iset] = int(p.get_entry("NumMembers"))-1
    print("nrep =", nrep[iset])
    # Arrays to store LHAPDF results
    if(iset==0):
        fit1 = np.zeros((nrep[iset],nsf,nq))
        fit1_cv = np.zeros((nsf,nq))
    if(iset==1):
        fit2 = np.zeros((nrep[iset],nsf,nq))
        fit2_cv = np.zeros((nsf,nq))
    if(iset==2):
        fit3 = np.zeros((nrep[iset],nsf,nq))
        fit3_cv = np.zeros((nsf,nq))
        
    # Run over replicas
    for i in range(1,nrep[iset]+1):
        p=lhapdf.mkPDF(pdfset[iset],i)

        # Run over x arrat
        for k in range(nq):
            
            q = Q[k]
          
            # run over flavours
            for isf in range(nsf):

                #----------------------------------------------------------------
                if(isf==0): # F2_nu_p
                    if(iset==0):
                        fit1[i-1][isf][k] =  2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
                    if(iset==1):
                        fit2[i-1][isf][k] =  2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
                    if(iset==2):
                        fit3[i-1][isf][k] = 2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
                #----------------------------------------------------------------
                #----------------------------------------------------------------
                if(isf==1): # F2_nubar_p
                    if(iset==0):
                        fit1[i-1][isf][k] =  2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                    if(iset==1):
                        fit2[i-1][isf][k] =   2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )                             
                    if(iset==2):
                        fit3[i-1][isf][k] =  2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                #----------------------------------------------------------------
                #----------------------------------------------------------------
                if(isf==2): # xF3_nu_p
                    if(iset==0):
                        fit1[i-1][isf][k] =  2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
                    if(iset==1):
                        fit2[i-1][isf][k] =  2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
                    if(iset==2):
                        fit3[i-1][isf][k] = 2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
                #----------------------------------------------------------------
                #----------------------------------------------------------------
                if(isf==3): # xF3_nubar_p
                    if(iset==0):
                        fit1[i-1][isf][k] =  2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                    if(iset==1):
                        fit2[i-1][isf][k] =   2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )                             
                    if(iset==2):
                        fit3[i-1][isf][k] =  2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                #----------------------------------------------------------------
                
    # Central values
    p=lhapdf.mkPDF(pdfset[iset],0)
    for k in range(nq):
        q = Q[k]
        
        for isf in range(nsf):


            #----------------------------------------------------------------
            if(isf==0): # F2_nu_p
                if(iset==0):
                    fit1_cv[isf][k] =  2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
                if(iset==1):
                    fit2_cv[isf][k] =  2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
                if(iset==2):
                    fit3_cv[isf][k] =  2*( p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) + p.xfxQ(-4,x,q) )
            #----------------------------------------------------------------
            #----------------------------------------------------------------
            if(isf==1): # F2_nubar_p
                if(iset==0):
                    fit1_cv[isf][k] =  2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                if(iset==1):
                    fit2_cv[isf][k] =  2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                if(iset==2):
                    fit3_cv[isf][k] =  2*( p.xfxQ(+2,x,q) + p.xfxQ(-1,x,q) + \
                                                   p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
            #----------------------------------------------------------------
            #----------------------------------------------------------------
            if(isf==2): # xF3_nu_p
                if(iset==0):
                    fit1_cv[isf][k] =  2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
                if(iset==1):
                    fit2_cv[isf][k] =  2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
                if(iset==2):
                    fit3_cv[isf][k] =  2*( -p.xfxQ(-2,x,q) + p.xfxQ(1,x,q) + \
                                                   p.xfxQ(3,x,q) - p.xfxQ(-4,x,q) )
            #----------------------------------------------------------------
            #----------------------------------------------------------------
            if(isf==3): # xF3_nubar_p
                if(iset==0):
                    fit1_cv[isf][k] =  2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                if(iset==1):
                    fit2_cv[isf][k] =  2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
                if(iset==2):
                    fit3_cv[isf][k] =  2*( p.xfxQ(+2,x,q) - p.xfxQ(-1,x,q) \
                                                  - p.xfxQ(-3,x,q) + p.xfxQ(+4,x,q) )
            #----------------------------------------------------------------
            
#---------------------------------------------------------------------
# Compute central values and uncertainties
#---------------------------------------------------------------------

for iset in range(nset):

    # MC PDF sets, 68% CL intervals
    if(error_option[iset]=="mc_68cl"):

        if(iset==0):
            p1_high = np.nanpercentile(fit1,84,axis=0)
            p1_low = np.nanpercentile(fit1,16,axis=0)
            p1_mid = ( p1_high + p1_low )/2.
            p1_error = ( p1_high - p1_low )/2.
        elif(iset==1):
            p2_high = np.nanpercentile(fit2,84,axis=0)
            p2_low = np.nanpercentile(fit2,16,axis=0)
            p2_mid = ( p2_high + p2_low )/2.
            p2_error = ( p2_high - p2_low )/2.
        elif(iset==2):
            p3_high = np.nanpercentile(fit3,84,axis=0)
            p3_low = np.nanpercentile(fit3,16,axis=0)
            p3_mid = ( p3_high + p3_low )/2.
            p3_error = ( p3_high - p3_low )/2.

    # MC PDF sets, 99% CL intervals (all replicas)
    elif(error_option[iset]=="mc_99cl"):

        if(iset==0):
            p1_high = np.nanpercentile(fit1,99.5,axis=0)
            p1_low = np.nanpercentile(fit1,0.5,axis=0)
            p1_mid = np.median(fit1,axis=0)
            p1_error = ( p1_high - p1_low )/2.
        elif(iset==1):
            p2_high = np.nanpercentile(fit2,99.5,axis=0)
            p2_low = np.nanpercentile(fit2,0.5,axis=0)
            p2_mid = ( p2_high + p2_low )/2.
            p2_mid = np.median(fit2,axis=0)
            p2_error = ( p2_high - p2_low )/2.
        elif(iset==2):
            p3_high = np.nanpercentile(fit3,99.5,axis=0)
            p3_low = np.nanpercentile(fit3,0.5,axis=0)
            p3_mid = ( p3_high + p3_low )/2.
            p3_error = ( p3_high - p3_low )/2.
        else:
            print("invalid option")
            exit()

    # MC PDF sets, one-sigma and mean
    elif(error_option[iset]=="mc_1sigma"):

        if(iset==0):
            p1_high = np.mean(fit1,axis=0) + np.std(fit1,axis=0)
            p1_low = np.mean(fit1,axis=0) - np.std(fit1,axis=0)
            p1_mid = np.mean(fit1,axis=0)
            p1_error= np.std(fit1,axis=0)
        elif(iset==1):
            p2_high = np.mean(fit2,axis=0) + np.std(fit2,axis=0)
            p2_low = np.mean(fit2,axis=0) - np.std(fit2,axis=0)
            p2_mid = np.mean(fit2,axis=0)
            p2_error= np.std(fit2,axis=0)
        elif(iset==2):
            p3_high = np.mean(fit3,axis=0) + np.std(fit3,axis=0)
            p3_low = np.mean(fit3,axis=0) - np.std(fit3,axis=0)
            p3_mid = np.mean(fit3,axis=0)
            p3_error= np.std(fit3,axis=0)
        else:
            print("invalid option")
            exit()

    # CT: asymmetric Hessian with then normalisation to one-sigma
    elif(error_option[iset]=="ct" or error_option[iset]=="mmht" ):

        if(iset==0):
            p1_mid = np.mean(fit1,axis=0)
            p1_error = np.std(fit1,axis=0)
            neig = int(nrep[iset]/2) # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p1_mid[isf][ix]=fit1_cv[isf][ix]
                    p1_error[isf][ix]=0 # initialisation
                    for ieg in range(neig):
                        #print(2*ieg+1,nrep[iset])
                        p1_error[isf][ix] = p1_error[isf][ix] \
                            + (fit1[2*ieg+1][isf][ix] - fit1[2*ieg][isf][ix] )**2.0
                    p1_error[isf][ix]=math.sqrt(p1_error[isf][ix])/2
                    if(error_option[iset]=="ct"):
                        p1_error[isf][ix]=p1_error[isf][ix] / 1.642 # from 90% to 68% CL
            p1_high = p1_mid + p1_error 
            p1_low = p1_mid - p1_error
        elif(iset==1):
            p2_mid = np.mean(fit2,axis=0)
            p2_error = np.std(fit2,axis=0)
            neig = int(nrep[iset]/2) # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p2_mid[isf][ix]=fit2_cv[isf][ix]
                    p2_error[isf][ix]=0 # initialisation
                    for ieg in range(neig):
                        #print(2*ieg+1,nrep[iset])
                        p2_error[isf][ix] = p2_error[isf][ix] \
                            + (fit2[2*ieg+1][isf][ix] - fit2[2*ieg][isf][ix] )**2.0
                    p2_error[isf][ix]=math.sqrt(p2_error[isf][ix])/2
                    if(error_option[iset]=="ct"):
                        p2_error[isf][ix]=p2_error[isf][ix] / 1.642 # from 90% to 68% CL
            p2_high = p2_mid + p2_error 
            p2_low = p2_mid - p2_error
        elif(iset==2):
            p3_mid = np.mean(fit3,axis=0)
            p3_error = np.std(fit3,axis=0)
            neig = int(nrep[iset]/2) # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p3_mid[isf][ix]=fit3_cv[isf][ix]
                    p3_error[isf][ix]=0 # initialisation
                    for ieg in range(neig):
                        p3_error[isf][ix] = p3_error[isf][ix] \
                            + (fit3[2*ieg+1][isf][ix] - fit3[2*ieg][isf][ix] )**2.0
                    p3_error[isf][ix]=math.sqrt(p3_error[isf][ix])/2
                    if(error_option[iset]=="ct"):
                        p3_error[isf][ix]=p3_error[isf][ix] / 1.642 # from 90% to 68% CL
            p3_high = p3_mid + p3_error 
            p3_low = p3_mid - p3_error
        else:
            print("invalid option")
            exit()

    # HERAPDF: symmetric Hessian
    elif(error_option[iset]=="symmhessian"):

        if(iset==0):
            p1_mid = np.mean(fit1,axis=0)
            p1_error = np.std(fit1,axis=0)
            neig =  int(nrep[iset]) # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p1_mid[isf][ix]=fit1_cv[isf][ix] # Central Hessian value
                    p1_error[isf][ix]=0 # initialisation
                    for ieg in range(neig):
                        p1_error[isf][ix] = p1_error[isf][ix] \
                            + (fit1[ieg][isf][ix] - p1_mid[isf][ix] )**2.0
                    p1_error[isf][ix]=math.sqrt(p1_error[isf][ix])
            p1_high = p1_mid + p1_error 
            p1_low = p1_mid - p1_error

        else:
            print("invalid option")
            exit()
       
    else:
        print("Invalid error option = ",error_option[iset])
        exit()
    
#----------------------------------------------------------------------

#*****************************************************************************
#*****************************************************************************

print("\n ****** Plotting absolute Structure Functions ******* \n")

ncols,nrows=2,2
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']
# x = 0.0126
#yranges=[[1.3,3.3],[1.3,3.3],[-0.2,0.7],[-0.2,0.7]]
# x = 0.25
yranges=[[0.55,1.05],[1.2,1.6],[0.45,0.8],[0.8,1.5]]
labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])
    p1=ax.plot(Q,p1_mid[isf],ls="dashed")
    ax.fill_between(Q,p1_high[isf],p1_low[isf],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    p3=ax.plot(Q,p2_mid[isf],ls="solid")
    ax.fill_between(Q,p2_high[isf],p2_low[isf],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
    p5=ax.plot(Q,p3_mid[isf],ls="dashdot")
    ax.fill_between(Q,p3_high[isf],p3_low[isf],color=rescolors[2],alpha=0.2)
    p6=ax.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)
    ax.set_xscale('linear')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=21)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=20)
    if(isf==0):
        #ax.text(0.65,0.85,r'$x=0.0126$',fontsize=16,transform=ax.transAxes)
        ax.text(0.65,0.85,r'$x=0.25$',fontsize=16,transform=ax.transAxes)
    

    # Add Genie
    if(isf==0):
        p7=ax.plot(genie_sf_q, genie_sf_f2,ls="dotted")
    if(isf==2):
        p7=ax.plot(genie_sf_q, genie_sf_f3,ls="dotted")

   

    # Add Yadism
    if(isf==0):
        p8=ax.plot(yadism_sf_q, yadism_f2,ls="dashdot")
        p9=ax.plot(yadism_sf_q, yadism_nnlo_f2,ls="solid")
    if(isf==2):
        p8=ax.plot(yadism_sf_q, yadism_f3,ls="dashdot")
        p9=ax.plot(yadism_sf_q, yadism_nnlo_f3,ls="solid")

    if(isf==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0]),p7[0],p8[0],p9[0]],\
                  [pdfsetlab[0],pdfsetlab[1],pdfsetlab[2],\
                   r"${\rm Bodek~Yang~(GENIE)}$",r"${\rm LO~YADISM+NNPDF4.0NNLO}$",r"${\rm NNLO~YADISM+NNPDF4.0NNLO}$"], \
                  frameon="True",loc=2,prop={'size':8})


py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction'+filelabel+'.pdf')
print('output plot: StructureFunction'+filelabel+'.pdf')

exit()
    
print("\n ****** Plotting ratios Structure Functions ******* \n")

# x = 0.0126
yranges=[[0.6,1.6],[0.5,1.5],[0.3,2.6],[-1.0,2.9]]
# x = 0.25
#yranges=[[0.6,1.5],[0.5,1.5],[0.5,1.5],[0.5,1.5]]
labelpdf=[r"$F_2^{\nu p}(x,Q)/F_2^{\rm (ref)}$",r"$F_2^{\bar{\nu} p}(x,Q)/F_2^{\rm (ref)}$",\
          r"$xF_3^{\nu p}(x,Q)/xF_3^{\rm (ref)}$",r"$xF_3^{\bar{\nu} p}(x,Q)/xF_3^{\rm (ref)}$"]

for isf in range(nsf):

    norm = p1_mid[isf]
    
    ax = py.subplot(gs[4+isf])
    p1=ax.plot(Q,p1_mid[isf]/norm,ls="dashed")
    ax.fill_between(Q,p1_high[isf]/norm,p1_low[isf]/norm,color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    p3=ax.plot(Q,p2_mid[isf]/norm,ls="solid")
    ax.fill_between(Q,p2_high[isf]/norm,p2_low[isf]/norm,color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
    p5=ax.plot(Q,p3_mid[isf]/norm,ls="dashdot")
    ax.fill_between(Q,p3_high[isf]/norm,p3_low[isf]/norm,color=rescolors[2],alpha=0.2)
    p6=ax.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)
    ax.set_xscale('linear')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=21)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=20)
    ax.text(0.05,0.85,r'$x=0.0126$',fontsize=16,transform=ax.transAxes)
    #ax.text(0.05,0.85,r'$x=0.25$',fontsize=16,transform=ax.transAxes)

    # Add Genie
    if(isf==0):
        p7=ax.plot(genie_sf_q, genie_sf_f2/norm,ls="dotted")
    if(isf==2):
        p7=ax.plot(genie_sf_q, genie_sf_f3/norm,ls="dotted")
    

    if(isf==0 or isf==2):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0]),p7[0]],\
                  [pdfsetlab[0],pdfsetlab[1],pdfsetlab[2],r"${\rm Bodek~Yang}$"], \
                  frameon="True",loc=1,prop={'size':13})

    
    if(isf==1 or isf==3):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0]),(p5[0],p6[0])],\
                  [pdfsetlab[0],pdfsetlab[1],pdfsetlab[2]], \
                  frameon="True",loc=1,prop={'size':13})

        
            
    
py.tight_layout(pad=1, w_pad=2, h_pad=1.0)
py.savefig('StructureFunction'+filelabel+'.pdf')
print('output plot: StructureFunction'+filelabel+'.pdf')

exit()

