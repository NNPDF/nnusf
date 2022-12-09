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
qmin = 0.3
qmax=300
# set x grid
Q = np.logspace(np.log10(qmin),np.log10(qmax),nq)

print(Q)

# Number of PDF sets to be compared
nset =2
# ----------------------------------------------------------------
# modernPDFs
#pdfset=["CT18NNLO","NNPDF40_nnlo_as_01180","MSHT20nnlo_as118"]
#pdfsetlab=[r"${\rm CT18}$",r"${\rm NNPDF4.0}$",r"${\rm MSHT20}$"]
# error_option=["ct","mc_68cl","mmht"]

# oldPDFs
pdfset=["NNPDF40_nnlo_as_01180","GRV98lo_patched"]
pdfsetlab=[r"${\rm LO~SFs+NNPDF4.0}$",r"${\rm LO~SFs+GRV98LO}$"]
error_option=["mc_68cl","ct"]

#----------------------------------------------
#----------------------------------------------
# Value of x
#x = 0.25
x = 0.0126
#----------------------------------------------
#----------------------------------------------

# FILELABEL
if( x > 0.0125 and x < 0.0127):
    filelabel="-allcomp_x0p0126"
if( x > 0.24 and x < 0.26):
    filelabel="-allcomp_x0p25"

#---------------------------
#---------------------------
print("\n Reading the GENIE structure functions \n")

# Read genie inputs
if( x > 0.0125 and x < 0.0127):
    genie_sf=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-BodekYang_x_0p0126.txt")
    genie_sf_nubar=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-nubar-BodekYang_x_0p0126.txt")
if( x > 0.24 and x < 0.26):
    genie_sf=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-BodekYang_x0p25.txt")
    genie_sf_nubar=np.loadtxt("Genie_Data/BodekYang/Genie-F2-xF3-nubar-BodekYang_x_0p25.txt")

print(genie_sf)



nq2_genie=111
genie_sf_q=np.zeros(nq2_genie)
genie_sf_f2=np.zeros(nq2_genie)
genie_sf_f3=np.zeros(nq2_genie)
genie_sf_f2_nubar=np.zeros(nq2_genie)
genie_sf_f3_nubar=np.zeros(nq2_genie)

for iq2 in range(nq2_genie):
    genie_sf_q[iq2] = math.sqrt( genie_sf[iq2][1] )
    genie_sf_f2[iq2] = genie_sf[iq2][2]
    genie_sf_f3[iq2] = x* genie_sf[iq2][3] # Genie gives F3 instead of xF3
    genie_sf_f2_nubar[iq2] = genie_sf_nubar[iq2][2]
    genie_sf_f3_nubar[iq2] = genie_sf_nubar[iq2][3] # Genie gives F3 instead of xF3

Q = genie_sf_q     
nq = Q.size

print(genie_sf_q)    
print(genie_sf_f2)
print(genie_sf_f3)
print(genie_sf_f2_nubar)
print(genie_sf_f3_nubar)


#
# BGR18
#
# Read genie inputs
if( x > 0.0125 and x < 0.0127):
    genie_sf_bgr=np.loadtxt("Genie_Data/BGR18/Genie-F2-xF3-BGR18_x0p0126.txt")
    genie_sf_bgr_nubar=np.loadtxt("Genie_Data/BGR18/Genie-F2-xF3-nubar-BGR18_x0p0126.txt")
if( x > 0.24 and x < 0.26):
    genie_sf_bgr=np.loadtxt("Genie_Data/BGR18/Genie-F2-xF3-BGR18_x0p25.txt")
    genie_sf_bgr_nubar=np.loadtxt("Genie_Data/BGR18/Genie-F2-xF3-nubar-BGR18_x0p25.txt")

genie_bgr_sf_q=np.zeros(nq2_genie)
genie_bgr_sf_f2=np.zeros(nq2_genie)
genie_bgr_sf_f3=np.zeros(nq2_genie)
genie_bgr_sf_f2_nubar=np.zeros(nq2_genie)
genie_bgr_sf_f3_nubar=np.zeros(nq2_genie)

for iq2 in range(nq2_genie):
    genie_bgr_sf_q[iq2] = math.sqrt( genie_sf_bgr[iq2][1] )
    genie_bgr_sf_f2[iq2] = genie_sf_bgr[iq2][2]
    genie_bgr_sf_f3[iq2] = genie_sf_bgr[iq2][3]
    genie_bgr_sf_f2_nubar[iq2] = genie_sf_bgr_nubar[iq2][2]
    genie_bgr_sf_f3_nubar[iq2] = -genie_sf_bgr_nubar[iq2][3] 
    
#print(genie_sf_q)    
print(genie_bgr_sf_f3)
print(genie_bgr_sf_f3_nubar)
#exit()
#print(genie_sf_f3)


#-------------------------------------------------------------
#-------------------------------------------------------------
print("\n *****************************************************************")
print("\n Reading the NNSF machine learning structure functions \n")
print(" *****************************************************************\n")

if( x > 0.0125 and x < 0.0127):
    nnsf_data = np.loadtxt("NNSF_data/Q2dep/NNSF_A1_x_0p0126.txt")
if( x > 0.24 and x < 0.26):
    nnsf_data = np.loadtxt("NNSF_data/Q2dep/NNSF_A1_x_0p25.txt")

print("nnsf data read correctly")

icount = 0
nrep_nnsf = int(76)
nq2_nnsf = int(150)
q2_nnsf = np.zeros(nq2_nnsf)
q_nnsf = np.zeros(nq2_nnsf)

nnsf_f2nu = np.zeros([nq2_nnsf, nrep_nnsf])
nnsf_f2nubar = np.zeros([nq2_nnsf, nrep_nnsf])
nnsf_xf3nu = np.zeros([nq2_nnsf, nrep_nnsf])
nnsf_xf3nubar = np.zeros([nq2_nnsf, nrep_nnsf])

for irep in range(nrep_nnsf):
    for iq2 in range(nq2_nnsf):
        xtmp = nnsf_data[icount][1]
        if (abs((xtmp - x)/x) > 1e-3):
            print("Incorrect value of x!")
            print("x = ",xtmp)
            exit()

        q2tmp = nnsf_data[icount][2]
        if(q2tmp < 0.09 or q2tmp > 100):
            print("Incorrect value of q2!")
            print("q2 = ",q2tmp)
            exit()
        q2_nnsf[iq2] = q2tmp
        q_nnsf[iq2] = pow(q2tmp,0.5)

        irep_tmp = int(nnsf_data[icount][0])
        #print(irep, " ",irep_tmp)
        if(irep_tmp != irep):
            print("Incorrect value of irep!")
            print("irep = ",irep_tmp)
            exit()

        nnsf_f2nu[iq2][irep] = nnsf_data[icount][3]
        nnsf_f2nubar[iq2][irep] = nnsf_data[icount][6]
        nnsf_xf3nu[iq2][irep] = nnsf_data[icount][5]
        nnsf_xf3nubar[iq2][irep] = nnsf_data[icount][8]

        icount = icount +1

#print(nnsf_f2nubar[69][75],nnsf_xf3nubar[69][75] )

# Evaluate median and 68%CL intervals
nnsf_f2nu_high = np.nanpercentile(nnsf_f2nu,84,axis=1)
nnsf_f2nu_low  = np.nanpercentile(nnsf_f2nu,16,axis=1)
nnsf_f2nu_mid = ( nnsf_f2nu_high + nnsf_f2nu_low )/2.
nnsf_f2nu_error = ( nnsf_f2nu_high - nnsf_f2nu_low )/2.

#print (nnsf_f2nu_mid.size)

nnsf_f2nubar_high = np.nanpercentile(nnsf_f2nubar,84,axis=1)
nnsf_f2nubar_low  = np.nanpercentile(nnsf_f2nubar,16,axis=1)
nnsf_f2nubar_mid = ( nnsf_f2nubar_high + nnsf_f2nubar_low )/2.
nnsf_f2nubar_error = ( nnsf_f2nubar_high - nnsf_f2nubar_low )/2.

nnsf_xf3nu_high = np.nanpercentile(nnsf_xf3nu,84,axis=1)
nnsf_xf3nu_low  = np.nanpercentile(nnsf_xf3nu,16,axis=1)
nnsf_xf3nu_mid = ( nnsf_xf3nu_high + nnsf_xf3nu_low )/2.
nnsf_xf3nu_error = ( nnsf_xf3nu_high - nnsf_xf3nu_low )/2.

nnsf_xf3nubar_high = np.nanpercentile(nnsf_xf3nubar,84,axis=1)
nnsf_xf3nubar_low  = np.nanpercentile(nnsf_xf3nubar,16,axis=1)
nnsf_xf3nubar_mid = ( nnsf_xf3nubar_high + nnsf_xf3nubar_low )/2.
nnsf_xf3nubar_error = ( nnsf_xf3nubar_high - nnsf_xf3nubar_low )/2.

print(q2_nnsf)
print(nnsf_f2nu_mid)
print(nnsf_f2nu_error)
print(nnsf_xf3nu_mid)
print(nnsf_xf3nu_error)

print("nnsf data processed correctly")


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

print("\n ****** Plotting absolute Structure Functions (Validation) ******* \n")

ncols,nrows=2,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if( x > 0.0125 and x < 0.0127):
    yranges=[[0,3.3],[-0.12,0.30]]
if( x > 0.24 and x < 0.26):
    yranges=[[0.53,0.80],[1.07,1.5],[0.38,0.65],[0.9,1.25]]
    
labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(2):

    ax = py.subplot(gs[isf])
  
    # GENIE BY
    if(isf==0):
        p1=ax.plot(genie_sf_q, genie_sf_f2,ls="solid",color=rescolors[0])
    if(isf==1):
        p1=ax.plot(genie_sf_q, genie_sf_f3_nubar,ls="solid",color=rescolors[0])
        
    ax.set_xscale('log')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=15)
    if( x > 0.0125 and x < 0.0127):
        if(isf==0):
            ax.text(0.03,0.90,r'$x=0.0126$',fontsize=14,transform=ax.transAxes)
    if( x > 0.24 and x < 0.26):
        if(isf==0):
            ax.text(0.65,0.85,r'$x=0.25$',fontsize=17,transform=ax.transAxes)

    plt.axvline(6,lw=1.5,ls="dashed",color=rescolors[3])
    ax.text(0.46,0.90,r'$Q_{\rm dat}$',fontsize=15,transform=ax.transAxes,color=rescolors[3])
    plt.axvline(40,lw=2,ls="dotted",color=rescolors[2])
    ax.text(0.74,0.90,r'$Q_{\rm thres}$',fontsize=15,transform=ax.transAxes,color=rescolors[2])

    if(isf==0):
        ax.text(0.49,0.20,r'$\rm  ML~fitted$',fontsize=12,transform=ax.transAxes,color=rescolors[3])
        ax.text(0.50,0.12,r'$\rm to~pQCD$',fontsize=12,transform=ax.transAxes,color=rescolors[3])
        ax.text(0.21,0.20,r'$\rm ML~fitted$',fontsize=12,transform=ax.transAxes,color="black")
        ax.text(0.21,0.12,r'$\rm to~SF\,data$',fontsize=12,transform=ax.transAxes,color="black")
        ax.text(0.79,0.16,r'$\rm pQCD$',fontsize=12,transform=ax.transAxes,color=rescolors[2])
        ax.arrow(x=5, y=1, dx=-4.2, dy=0, width=.05)
        ax.arrow(x=45, y=1, dx=105, dy=0, width=.05,head_length=50)
        ax.arrow(x=7.5, y=1, dx=20, dy=0, width=.05,head_length=7)

    if(isf==0 or isf==1):
        ax.text(0.12,0.56,r'$\rm Region\,I$',fontsize=15,transform=ax.transAxes,color="black")
        ax.text(0.47,0.56,r'$\rm Region\,II$',fontsize=15,transform=ax.transAxes,color=rescolors[3])
        ax.text(0.75,0.56,r'$\rm Region\,III$',fontsize=15,transform=ax.transAxes,color=rescolors[2])

        
    
    
    if(isf==1):
        ax.legend([p1[0]],
                  [r"${\rm Bodek\,Yang}$"], \
                  frameon="True",loc=3,prop={'size':12})

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Scheme'+filelabel+'.pdf')
print('output plot: StructureFunction-Scheme'+filelabel+'.pdf')

#################################################################################
#################################################################################



exit()
