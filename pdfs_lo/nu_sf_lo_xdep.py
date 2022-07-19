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
nx = 200
xmin = 3e-3
xmax=1.0
# set x grid
X = np.logspace(np.log10(xmin),np.log10(xmax),nx)
print(X)

# Number of PDF sets to be compared
nset =2
# ----------------------------------------------------------------
# modernPDFs
#pdfset=["CT18NNLO","NNPDF40_nnlo_as_01180","MSHT20nnlo_as118"]
#pdfsetlab=[r"${\rm CT18}$",r"${\rm NNPDF4.0}$",r"${\rm MSHT20}$"]
# error_option=["ct","mc_68cl","mmht"]

# oldPDFs
pdfset=["NNPDF40_nnlo_as_01180","GRV98lo_patched"]
pdfsetlab=[r"${\rm LO~SF+NNPDF4.0NNLO}$",r"${\rm LO~SF+GRV98LO}$"]
error_option=["mc_68cl","ct"]
filelabel="-allcomp_q2gev"
#filelabel="-allcomp_q10gev"

#----------------------------------------------
#----------------------------------------------
# Value of Q
q = 2 # gev
#q = 10 # gev
#----------------------------------------------
#----------------------------------------------

#---------------------------
#---------------------------
print("\n Reading the GENIE structure functions \n")

# Read genie inputs
# Bodek-Yang model
# Neutrino structure functions F2 and F3 on free protons
genie_sf_nu_p=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-BodekYang.txt")
# Anti-Neutrino structure functions F2 and F3 on free protons
genie_sf_nubar_p=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-nubar-BodekYang.txt")

nq2_genie=111
nx_genie =101
genie_sf_x=np.zeros(nx_genie)
genie_sf_f2=np.zeros(nx_genie)
genie_sf_f3=np.zeros(nx_genie)
genie_sf_f2_nubar=np.zeros(nx_genie)
genie_sf_f3_nubar=np.zeros(nx_genie)

#iq2_genie=30  # Q = 10 GeV
iq2_genie=16  # Q = 2 GeV

# Check
q_check=math.pow(genie_sf_nu_p[iq2_genie][1],0.5)
reldiff=abs( (q_check-q) /q )
if(reldiff > 0.05):
    print("Q mismatch")
    print(q_check," ",q)
    exit()

icount=0
for ix in range(nx_genie):
    index = icount*nq2_genie+iq2_genie
    #print(ix," ",index," ",math.pow(genie_sf_nu_p[index][1],0.5)," ",genie_sf_nu_p[index][0])
    genie_sf_x[ix] = genie_sf_nu_p[index][0]
    genie_sf_f2[ix] = genie_sf_nu_p[index][2]
    genie_sf_f3[ix] = genie_sf_x[ix]* genie_sf_nu_p[index][3] # Genie gives F3 instead of xF3
    genie_sf_f2_nubar[ix] = genie_sf_nubar_p[index][2]
    genie_sf_f3_nubar[ix] = genie_sf_nubar_p[index][3] # Genie gives F3 instead of xF3 ??
    icount = icount+1


# Read genie inputs
# BGR18 calculation
# Neutrino structure functions F2 and F3 on free protons
genie_sf_nu_p_bgr=np.loadtxt("Genie_Data/BGR18/BGR_nu_free_p.txt")
genie_sf_nub_p_bgr=np.loadtxt("Genie_Data/BGR18/BGR_nub_free_p.txt")

genie_sf_bgr_x=np.zeros(nx_genie)
genie_sf_bgr_f2=np.zeros(nx_genie)
genie_sf_bgr_f3=np.zeros(nx_genie)
genie_sf_bgr_f2_nub=np.zeros(nx_genie)
genie_sf_bgr_f3_nub=np.zeros(nx_genie)

icount=0
for ix in range(nx_genie):
    index = icount*nq2_genie+iq2_genie
    #print(ix," ",index," ",math.pow(genie_sf_nu_p[index][1],0.5)," ",genie_sf_nu_p[index][0])
    genie_sf_bgr_x[ix] = genie_sf_nu_p_bgr[index][0]
    genie_sf_bgr_f2[ix] = genie_sf_nu_p_bgr[index][2]
    genie_sf_bgr_f3[ix] = genie_sf_nu_p_bgr[index][3]
    genie_sf_bgr_f2_nub[ix] = genie_sf_nub_p_bgr[index][2]
    genie_sf_bgr_f3_nub[ix] = genie_sf_nub_p_bgr[index][3] 
    icount = icount+1

#---------------------------
#---------------------------

#---------------------------
#---------------------------
print("\n Reading the YADISM structure functions \n")

# Neutrino structure functions F2 and F3 on free protons
yadism_f2_lo_nu_p=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F2.txt")
yadism_f3_lo_nu_p=np.loadtxt("Yadism_data/LO_NNPDF40_yadism/F3.txt")
yadism_f2_nnlo_nu_p=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F2.txt")
yadism_f3_nnlo_nu_p=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism/F3.txt")

# Neutrino structure functions F2 and F3 on free protons
yadism_f2_lo_nubar_p=np.loadtxt("Yadism_data/LO_NNPDF40_yadism_nubar/F2.txt")
yadism_f3_lo_nubar_p=np.loadtxt("Yadism_data/LO_NNPDF40_yadism_nubar/F3.txt")
yadism_f2_nnlo_nubar_p=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism_nubar/F2.txt")
yadism_f3_nnlo_nubar_p=np.loadtxt("Yadism_data/NNLO_NNPDF40_yadism_nubar/F3.txt")

nq2_yadism=20
nx_yadism =30
yadism_sf_x=np.zeros(nx_yadism)
yadism_sf_f2_lo=np.zeros(nx_yadism)
yadism_sf_f3_lo=np.zeros(nx_yadism)
yadism_sf_f2_nnlo=np.zeros(nx_yadism)
yadism_sf_f3_nnlo=np.zeros(nx_yadism)
yadism_sf_f2_lo_nubar=np.zeros(nx_yadism)
yadism_sf_f3_lo_nubar=np.zeros(nx_yadism)
yadism_sf_f2_nnlo_nubar=np.zeros(nx_yadism)
yadism_sf_f3_nnlo_nubar=np.zeros(nx_yadism)

#iq2_yadism=16  # Q = 10 GeV
iq2_yadism=2  # Q = 2 GeV

# Check
q_check=math.pow(yadism_f2_lo_nu_p[iq2_yadism][2],0.5)
reldiff=abs( (q_check-q) /q )
if(reldiff > 0.05):
    print("Q mismatch")
    print(q_check," ",q)
    exit()

icount=0
for ix in range(nx_yadism):
    index = icount*nq2_yadism+iq2_yadism
    print(ix," ",index," ",math.pow(yadism_f2_lo_nu_p[index][1],0.5)," ",yadism_f2_lo_nu_p[index][2])
    yadism_sf_x[ix] = yadism_f2_lo_nu_p[index][1]
    yadism_sf_f2_lo[ix] = yadism_f2_lo_nu_p[index][3]
    yadism_sf_f3_lo[ix] = yadism_f3_lo_nu_p[index][3]
    yadism_sf_f2_nnlo[ix] = yadism_f2_nnlo_nu_p[index][3]
    yadism_sf_f3_nnlo[ix] = yadism_f3_nnlo_nu_p[index][3]
    yadism_sf_f2_lo_nubar[ix] = yadism_f2_lo_nubar_p[index][3]
    yadism_sf_f3_lo_nubar[ix] = yadism_f3_lo_nubar_p[index][3]
    yadism_sf_f2_nnlo_nubar[ix] = yadism_f2_nnlo_nubar_p[index][3]
    yadism_sf_f3_nnlo_nubar[ix] = yadism_f3_nnlo_nubar_p[index][3] 
    icount = icount+1

#print(yadism_sf_f3_lo)

#*****************************************************************
#*****************************************************************
#*****************************************************************


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
        fit1 = np.zeros((nrep[iset],nsf,nx))
        fit1_cv = np.zeros((nsf,nx))
    if(iset==1):
        fit2 = np.zeros((nrep[iset],nsf,nx))
        fit2_cv = np.zeros((nsf,nx))
    if(iset==2):
        fit3 = np.zeros((nrep[iset],nsf,nx))
        fit3_cv = np.zeros((nsf,nx))
        
    # Run over replicas
    for i in range(1,nrep[iset]+1):
        p=lhapdf.mkPDF(pdfset[iset],i)

        # Run over x arrat
        for k in range(nx):

            x = X[k]
                     
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
    for k in range(nx):
        x = X[k]
        
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
            for ix in range(nx):
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
            for ix in range(nx):
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
            for ix in range(nx):
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
            for ix in range(nx):
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
# Q = 10 GeV
#yranges=[[0,4.3],[0,4.3],[0,0.8],[-0.25,1.20]]
# Q = 2 GeV
yranges=[[0,2.3],[0,2.3],[0,0.8],[-0.25,1.30]]
labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    # LO SF + NNPDF4.0
    p1=ax.plot(X,p1_mid[isf],ls="dotted",lw=2)
    ax.fill_between(X,p1_high[isf],p1_low[isf],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    # YADISM LO
    if(isf==0):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_lo,ls="dashed",color=rescolors[1])
    if(isf==1):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_lo_nubar,ls="dashed",color=rescolors[1])
    if(isf==2):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_lo,ls="dashed",color=rescolors[1])
    if(isf==3):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_lo_nubar,ls="dashed",color=rescolors[1])
    
    # LO SF + GRV98
    p4=ax.plot(X,p2_mid[isf],ls="dotted",color=rescolors[2],lw=3)

    # GENIE BY
    if(isf==0):
        p5=ax.plot(genie_sf_x, genie_sf_f2,ls="solid",color=rescolors[3])
    if(isf==1):
        p5=ax.plot(genie_sf_x, genie_sf_f2_nubar,ls="solid",color=rescolors[3])
    if(isf==2):
        p5=ax.plot(genie_sf_x, genie_sf_f3,ls="solid",color=rescolors[3])
    if(isf==3):
        p5=ax.plot(genie_sf_x, genie_sf_f3_nubar,ls="solid",color=rescolors[3])

    # NNLO YADISM
    if(isf==0):
        p6=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo,ls="dashed",color=rescolors[4])
    if(isf==1):
        p6=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[4])
    if(isf==2):
        p6=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo,ls="dashed",color=rescolors[4])
    if(isf==3):
        p6=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[4])

    # GENIE BGR18
    if(isf==0):
        p7=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2,ls="dashdot",color=rescolors[5])
    if(isf==1):
        p7=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2_nub,ls="dashdot",color=rescolors[5])
    if(isf==2):
        p7=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f3,ls="dashdot",color=rescolors[5])
    if(isf==3):
        p7=ax.plot(genie_sf_bgr_x, (-1)*genie_sf_bgr_f3_nub,ls="dashdot",color=rescolors[5])
      
    ax.set_xscale('log')
    ax.set_xlim(xmin,xmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>1):
        ax.set_xlabel(r'$x$',fontsize=15)
    if(isf==0):
        #ax.text(0.67,0.85,r'$Q=10~{\rm GeV}$',fontsize=14,transform=ax.transAxes)
        ax.text(0.67,0.85,r'$Q=2~{\rm GeV}$',fontsize=15,transform=ax.transAxes)
 
    if(isf==1):
        ax.legend([(p1[0],p2[0]),p3[0],p4[0],p5[0],p6[0],p7[0]],\
                  [pdfsetlab[0],r"${\rm YADISM~(LO)+NNPDF4.0}$",\
                   pdfsetlab[1],\
                   r"${\rm Bodek~Yang~(GENIE)}$",\
                   r"${\rm YADISM~(NNLO)+NNPDF4.0}$",r"${\rm BGR18~(GENIE)}$"], \
                  frameon="True",loc=3,prop={'size':10})
        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-xdep'+filelabel+'.pdf')


#
# Now evaluate the K-factors with YADISM for fixed PDF
#
print("\n Evaluate the K-factors with YADISM \n")

print("\n ****** Plotting absolute Structure Functions ******* \n")

py.clf()
ncols,nrows=1,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']
# Q = 2 GeV

isf=0
ax = py.subplot(gs[isf])

# F2nu
p1=ax.plot(yadism_sf_x,  yadism_sf_f2_nnlo/yadism_sf_f2_lo,ls="solid",lw=2,color=rescolors[0])

# F2nubar
p2=ax.plot(yadism_sf_x,  yadism_sf_f2_nnlo_nubar/yadism_sf_f2_lo_nubar,ls="dashed",lw=2,color=rescolors[1])

# xF3
p3=ax.plot(yadism_sf_x,  yadism_sf_f3_nnlo/yadism_sf_f3_lo,ls="solid",lw=2,color=rescolors[2])

# xF3
p4=ax.plot(yadism_sf_x,  abs(yadism_sf_f3_nnlo_nubar/yadism_sf_f3_lo_nubar),ls="dashed",lw=2,color=rescolors[3])

ax.set_xscale('log')
ax.set_xlim(xmin,xmax)
ax.tick_params(which='both',direction='in',labelsize=12,right=True)
ax.tick_params(which='major',length=7)
ax.tick_params(which='minor',length=4)
ax.set_ylabel(r"${\rm NNLO/LO~}K\,{\rm factor}$",fontsize=17)
# Q = 10 GeV
ax.set_ylim(0.8,1.3)
# Q = 2 GeV
ax.set_ylim(0.2,1.8)
ax.set_xlabel(r'$x$',fontsize=15)
#ax.text(0.70,0.10,r'$Q=10~{\rm GeV}$',fontsize=14,transform=ax.transAxes)
ax.text(0.70,0.10,r'$Q=2~{\rm GeV}$',fontsize=14,transform=ax.transAxes)

ax.legend([p1[0],p2[0],p3[0],p4[0]],[r"$F_2^{\nu p}$",r"$F_2^{\bar{\nu} p}$",r"$xF_3^{\nu p}$",r"$xF_3^{\bar{\nu} p}$"], \
          frameon="True",loc=2,prop={'size':11})
        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-xdep-Kfact'+filelabel+'.pdf')
print('output plot: StructureFunction-xdep-Kfact'+filelabel+'.pdf')

exit()
