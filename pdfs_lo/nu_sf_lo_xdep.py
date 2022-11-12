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
xmin = 1e-3
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
pdfsetlab=[r"{\sc LO-SF-NNPDF4.0}",r"{\sc LO-SF-GRV98}"]
error_option=["mc_68cl","ct"]

#----------------------------------------------
#----------------------------------------------
# Value of Q
#q = 2 # gev
q = 10 # gev
#----------------------------------------------
#----------------------------------------------

if(q > 1.9 and q < 2.1):
    filelabel="-allcomp_q2gev"
if(q > 9.9 and q < 10.1):
    filelabel="-allcomp_q10gev"

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

if(q > 1.9 and q < 2.1):
    iq2_genie=16  # Q = 2 GeV
if(q > 9.9 and q < 10.1):
    iq2_genie=30  # Q = 10 GeV

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
#yadism_f2_lo_nu_p=np.loadtxt("Yadism_data_v1/LO_NNPDF40_yadism/F2.txt")
#yadism_f3_lo_nu_p=np.loadtxt("Yadism_data_v1/LO_NNPDF40_yadism/F3.txt")
#yadism_f2_nnlo_nu_p=np.loadtxt("Yadism_data_v1/NNLO_NNPDF40_yadism/F2.txt")
#yadism_f3_nnlo_nu_p=np.loadtxt("Yadism_data_v1/NNLO_NNPDF40_yadism/F3.txt")

yadism_f2_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/F2.txt")
yadism_f3_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/F3.txt")
yadism_fl_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/FL.txt")
yadism_f2_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/F2.txt")
yadism_f3_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/F3.txt")
yadism_fl_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/FL.txt")
yadism_f2_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/F2.txt")
yadism_f3_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/F3.txt")
yadism_fl_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/FL.txt")

# Neutrino structure functions F2 and F3 on free protons
#yadism_f2_lo_nubar_p=np.loadtxt("Yadism_data_v1/LO_NNPDF40_yadism_nubar/F2.txt")
#yadism_f3_lo_nubar_p=np.loadtxt("Yadism_data_v1/LO_NNPDF40_yadism_nubar/F3.txt")
#yadism_f2_nnlo_nubar_p=np.loadtxt("Yadism_data_v1/NNLO_NNPDF40_yadism_nubar/F2.txt")
#yadism_f3_nnlo_nubar_p=np.loadtxt("Yadism_data_v1/NNLO_NNPDF40_yadism_nubar/F3.txt")

yadism_f2_lo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/LO/predictions/F2.txt")
yadism_f3_lo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/LO/predictions/F3.txt")
yadism_fl_lo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/LO/predictions/FL.txt")
yadism_f2_nlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NLO/predictions/F2.txt")
yadism_f3_nlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NLO/predictions/F3.txt")
yadism_fl_nlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NLO/predictions/FL.txt")
yadism_f2_nnlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NNLO/predictions/F2.txt")
yadism_f3_nnlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NNLO/predictions/F3.txt")
yadism_fl_nnlo_nubar_p=np.loadtxt("Yadism_data_v2/antineutrino/NNLO/predictions/FL.txt")

nq2_yadism=20
nx_yadism =30
yadism_sf_x=np.zeros(nx_yadism)
yadism_sf_f2_lo=np.zeros(nx_yadism)
yadism_sf_f3_lo=np.zeros(nx_yadism)
yadism_sf_fl_lo=np.zeros(nx_yadism)
yadism_sf_f2_nlo=np.zeros(nx_yadism)
yadism_sf_f3_nlo=np.zeros(nx_yadism)
yadism_sf_fl_nlo=np.zeros(nx_yadism)
yadism_sf_f2_nnlo=np.zeros(nx_yadism)
yadism_sf_f3_nnlo=np.zeros(nx_yadism)
yadism_sf_fl_nnlo=np.zeros(nx_yadism)
yadism_sf_f2_lo_nubar=np.zeros(nx_yadism)
yadism_sf_f3_lo_nubar=np.zeros(nx_yadism)
yadism_sf_fl_lo_nubar=np.zeros(nx_yadism)
yadism_sf_f2_nlo_nubar=np.zeros(nx_yadism)
yadism_sf_f3_nlo_nubar=np.zeros(nx_yadism)
yadism_sf_fl_nlo_nubar=np.zeros(nx_yadism)
yadism_sf_f2_nnlo_nubar=np.zeros(nx_yadism)
yadism_sf_f3_nnlo_nubar=np.zeros(nx_yadism)
yadism_sf_fl_nnlo_nubar=np.zeros(nx_yadism)

if(q > 1.9 and q < 2.1):
    iq2_yadism=2  # Q = 2 GeV
if(q > 9.9 and q < 10.1):
    iq2_yadism=16  # Q = 10 GeV

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
    # Neutrino SFs
    yadism_sf_x[ix] = yadism_f2_lo_nu_p[index][1]
    yadism_sf_f2_lo[ix] = yadism_f2_lo_nu_p[index][3]
    yadism_sf_f3_lo[ix] = yadism_f3_lo_nu_p[index][3]
    yadism_sf_fl_lo[ix] = yadism_fl_lo_nu_p[index][3]
    yadism_sf_f2_nlo[ix] = yadism_f2_nlo_nu_p[index][3]
    yadism_sf_f3_nlo[ix] = yadism_f3_nlo_nu_p[index][3]
    yadism_sf_fl_nlo[ix] = yadism_fl_nlo_nu_p[index][3]
    yadism_sf_f2_nnlo[ix] = yadism_f2_nnlo_nu_p[index][3]
    yadism_sf_f3_nnlo[ix] = yadism_f3_nnlo_nu_p[index][3]
    yadism_sf_fl_nnlo[ix] = yadism_fl_nnlo_nu_p[index][3]
    # Anti-Neutrino SFs
    yadism_sf_f2_lo_nubar[ix] = yadism_f2_lo_nubar_p[index][3]
    yadism_sf_f3_lo_nubar[ix] = yadism_f3_lo_nubar_p[index][3]
    yadism_sf_fl_lo_nubar[ix] = yadism_fl_lo_nubar_p[index][3]
    yadism_sf_f2_nlo_nubar[ix] = yadism_f2_nlo_nubar_p[index][3]
    yadism_sf_f3_nlo_nubar[ix] = yadism_f3_nlo_nubar_p[index][3]
    yadism_sf_fl_nlo_nubar[ix] = yadism_fl_nlo_nubar_p[index][3]
    yadism_sf_f2_nnlo_nubar[ix] = yadism_f2_nnlo_nubar_p[index][3]
    yadism_sf_f3_nnlo_nubar[ix] = yadism_f3_nnlo_nubar_p[index][3]
    yadism_sf_fl_nnlo_nubar[ix] = yadism_fl_nnlo_nubar_p[index][3] 
    icount = icount+1

#print(yadism_sf_f3_lo)

#*****************************************************************
#*****************************************************************
#*****************************************************************

#-------------------------------------------------------------
#-------------------------------------------------------------
print("\n *****************************************************************")
print("\n Reading the NNSF machine learning structure functions \n")
print(" *****************************************************************\n")

nnsf_data = np.loadtxt("NNSF_data/xdep/NNSF_A1_xdep_q_2_10.txt")

print("nnsf data read correctly")

icount = 0
nrep_nnsf = int(76)
nx_nnsf = int(70)
x_nnsf = np.zeros(nx_nnsf)

nnsf_f2nu = np.zeros([nx_nnsf, nrep_nnsf])
nnsf_f2nubar = np.zeros([nx_nnsf, nrep_nnsf])
nnsf_xf3nu = np.zeros([nx_nnsf, nrep_nnsf])
nnsf_xf3nubar = np.zeros([nx_nnsf, nrep_nnsf])

# Loop over x
for ix in range(nx_nnsf):

    xtmp = nnsf_data[icount][1]
    if (xtmp < 0.009 or xtmp> 1.0):
        print("Incorrect value of x!")
        print("x = ",xtmp)
        exit()
    x_nnsf[ix] = xtmp
    
    # Loop over replicas
    for irep in range(nrep_nnsf):

        irep_tmp = int(nnsf_data[icount][0])
        #print(irep, " ",irep_tmp)
        if(irep_tmp != irep):
            print("Incorrect value of irep!")
            print("irep = ",irep_tmp)
            exit()

        # Loop over q2
        for iq2 in range(2):

            q2tmp = nnsf_data[icount][2]
            print("iq2, q2tmp = ",iq2," ",q2tmp)
            if(q2tmp < 0.09 or q2tmp > 100):
                print("Incorrect value of q2!")
                print("q2 = ",q2tmp)
                exit()
        
            if(iq2 == 0 ):
                diff=abs( ((q2tmp - 4.0 ) / q2tmp) )
                if(diff > 1e-5):
                    print("Incorrect value of q2 (not 4)!")
                    print("q2 = ",q2tmp)
                    exit()
            if(iq2 == 1 ):
                diff=abs( ((q2tmp - 100 ) / q2tmp) )
                if( diff > 1e-5):
                    print("Incorrect value of q2 (not 100)!")
                    print("q2 = ",q2tmp)
                    exit()

            # Fill the arrays
            if( ( 1.99 < q < 2.01 ) and ( iq2 ==0 )):
                nnsf_f2nu[ix][irep] = nnsf_data[icount][3]
                nnsf_f2nubar[ix][irep] = nnsf_data[icount][6]
                nnsf_xf3nu[ix][irep] = nnsf_data[icount][5]
                nnsf_xf3nubar[ix][irep] = nnsf_data[icount][8]
                diff=abs( ((q2tmp - 4.0 ) / q2tmp) )
                if(diff > 1e-5):
                    print("Incorrect value of q2 (not 4)!")
                    print("q2 = ",q2tmp)
                    exit()
            if( ( 9.99 < q < 10.01 ) and ( iq2 ==1 ) ):
                nnsf_f2nu[ix][irep] = nnsf_data[icount][3]
                nnsf_f2nubar[ix][irep] = nnsf_data[icount][6]
                nnsf_xf3nu[ix][irep] = nnsf_data[icount][5]
                nnsf_xf3nubar[ix][irep] = nnsf_data[icount][8]
                diff=abs( ((q2tmp - 100 ) / q2tmp) )
                if(diff > 1e-5):
                    print("Incorrect value of q2 (not 100)!")
                    print("q2 = ",q2tmp)
                    exit()

            # Increase counter
            icount = icount +1
            
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

print("\n x => \n")
print(x_nnsf)
print("\n f2nu_mid => \n")
print(nnsf_f2nu_mid)
print("\n f2nu_err => \n")
print(nnsf_f2nu_error)
print("\n xf3_mid => \n")
print(nnsf_xf3nu_mid)
print("\n xf3_err => \n")
print(nnsf_xf3nu_error)


print("nnsf data processed correctly")

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------



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

print("\n ****** Plotting absolute Structure Functions (Validation) ******* \n")

ncols,nrows=4,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']
if(q > 1.9 and q < 2.1):
    yranges=[[0,2.7],[0,2.7],[0,0.8],[-0.5,1.30]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,4.6],[0,4.6],[0,1.3],[-0.7,1.20]]
    
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
      
    ax.set_xscale('log')
    ax.set_xlim(xmin,xmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>(-1)):
        ax.set_xlabel(r'$x$',fontsize=15)
    if(isf==0):
        if(q > 9.9 and q < 10.1):
            ax.text(0.67,0.85,r'$Q=10~{\rm GeV}$',fontsize=16,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.67,0.90,r'$Q=2~{\rm GeV}$',fontsize=16,transform=ax.transAxes)
 
    if(isf==1):
        ax.legend([(p1[0],p2[0]),p3[0],p4[0],p5[0]],\
                  [pdfsetlab[0],r"{\sc YADISM-LO}",\
                   pdfsetlab[1],\
                   r"{\sc Bodek-Yang}"], \
                  frameon="True",loc=3,prop={'size':12})
        
py.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
py.savefig('StructureFunction-Validation-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-Validation-xdep'+filelabel+'.pdf')


#*****************************************************************************
#*****************************************************************************

print("\n ****** Plotting absolute Structure Functions (Perturbative) ******* \n")

ncols,nrows=2,2
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if(q > 1.9 and q < 2.1):
    yranges=[[0,2.3],[0,2.3],[0,0.8],[-0.25,1.30]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,4.6],[0,4.6],[0,1.3],[-0.7,1.20]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    # YADISM LO
    if(isf==0):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo,ls="dashed",color=rescolors[1])
    if(isf==1):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo_nubar,ls="dashed",color=rescolors[1])
    if(isf==2):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo,ls="dashed",color=rescolors[1])
    if(isf==3):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo_nubar,ls="dashed",color=rescolors[1])
    
    # GENIE BGR18
    if(isf==0):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2,ls="dashdot",color=rescolors[5])
    if(isf==1):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2_nub,ls="dashdot",color=rescolors[5])
    if(isf==2):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f3,ls="dashdot",color=rescolors[5])
    if(isf==3):
        p2=ax.plot(genie_sf_bgr_x, (-1)*genie_sf_bgr_f3_nub,ls="dashdot",color=rescolors[5])

    # NNLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo,ls="dashed",color=rescolors[4])
    if(isf==1):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[4])
    if(isf==2):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo,ls="dashed",color=rescolors[4])
    if(isf==3):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[4])
      
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
        if(q > 9.9 and q < 10.1):
            ax.text(0.67,0.85,r'$Q=10~{\rm GeV}$',fontsize=14,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.67,0.85,r'$Q=2~{\rm GeV}$',fontsize=15,transform=ax.transAxes)
            
 
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"${\rm YADISM~(LO)+NNPDF4.0}$",\
                   r"${\rm BGR18 NLO (NNPDF3.1)~{(GENIE)}}$",\
                   r"${\rm YADISM~(NNLO)+NNPDF4.0}$"],
                  frameon="True",loc=1,prop={'size':9})
        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Perturbative-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-Perturbative-xdep'+filelabel+'.pdf')

#*****************************************************************************
#*****************************************************************************

print("\n ****** Plotting absolute Structure Functions (Comparisons Pre-Fit) ******* \n")

ncols,nrows=4,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if(q > 1.9 and q < 2.1):
    yranges=[[0,2.7],[0,2.7],[0,0.7],[-0.25,1.25]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,5.0],[0,5.0],[0,1.3],[-0.7,1.25]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    # GENIE BY
    if(isf==0):
        p1=ax.plot(genie_sf_x, genie_sf_f2,ls="solid",color=rescolors[3])
    if(isf==1):
        p1=ax.plot(genie_sf_x, genie_sf_f2_nubar,ls="solid",color=rescolors[3])
    if(isf==2):
        p1=ax.plot(genie_sf_x, genie_sf_f3,ls="solid",color=rescolors[3])
    if(isf==3):
        p1=ax.plot(genie_sf_x, genie_sf_f3_nubar,ls="solid",color=rescolors[3])

    # GENIE BGR18
    if(isf==0):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2,ls="dashdot",color=rescolors[5])
    if(isf==1):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2_nub,ls="dashdot",color=rescolors[5])
    if(isf==2):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f3,ls="dashdot",color=rescolors[5])
    if(isf==3):
        p2=ax.plot(genie_sf_bgr_x, (-1)*genie_sf_bgr_f3_nub,ls="dashdot",color=rescolors[5])

    # NNLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo,ls="dashed",color=rescolors[4])
    if(isf==1):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[4])
    if(isf==2):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo,ls="dashed",color=rescolors[4])
    if(isf==3):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[4])
    
    ax.set_xscale('log')
    ax.set_xlim(xmin,xmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>-1):
        ax.set_xlabel(r'$x$',fontsize=15)
    if(isf==0):
        if(q > 9.9 and q < 10.1):
            ax.text(0.67,0.85,r'$Q=10~{\rm GeV}$',fontsize=14,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.67,0.85,r'$Q=2~{\rm GeV}$',fontsize=15,transform=ax.transAxes)
             
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"{\sc Bodek-Yang}",\
                   r"{\sc BGR18}",\
                   r"{\sc YADISM-NNLO}"],
                  frameon="True",loc=3,prop={'size':12})
                        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-ComparisonsPreFit-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-ComparisonsPreFit-xdep'+filelabel+'.pdf')


#*******************************************************************************
#*******************************************************************************
#
# Study perturbative stability by comparing yadism LO, NLO, NNLO for fixed PDF
# and evaluate the corresponding K-factors
# Also for FL, since now it is available
# Ratio to NLO as reference
#

print("\n ****** Plotting Perturbative Stability of Structure Functions  ******* \n")

ncols,nrows=3,4
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if(q > 1.9 and q < 2.1):
    yranges=[[0,2.8],[0,1.0],[0,0.8],[0.7,1.30],[0.5,2.5],[0,1.5],\
    [0,2.8],[-0.5,1.3],[0,0.8],[0.7,1.30],[0.5,2.0],[0,1.5]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,4.8],[0,0.8],[0,1.0],[0.7,1.20],[0.5,1.3],[0,1.2],\
    [0,4.8],[-0.8,1.5],[0,1.2],[0.7,1.20],[0.5,1.3],[0,1.2]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$xF_3^{\nu p}(x,Q)$",r"$F_L^{\nu p}(x,Q)$",\
          r"${\rm Ratio~to~NNLO}$",r"${\rm Ratio~to~NNLO}$",r"${\rm Ratio~to~NNLO}$",\
          r"$F_2^{\bar{\nu} p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$",r"$F_L^{\bar{\nu} p}(x,Q)$",\
          r"${\rm Ratio~to~NNLO}$",r"${\rm Ratio~to~NNLO}$",r"${\rm Ratio~to~NNLO}$"]

for isf in range(12):

    ax = py.subplot(gs[isf])

    if(isf==0):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f2_nlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo,ls="solid",color=rescolors[2])

    if(isf==1):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo,ls="solid",color=rescolors[2])

    if(isf==2):
        p1=ax.plot(yadism_sf_x, yadism_sf_fl_lo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_fl_nlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo,ls="solid",color=rescolors[2])

    if(isf==3):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo/yadism_sf_f2_nnlo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f2_nlo/yadism_sf_f2_nnlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo/yadism_sf_f2_nnlo,ls="solid",color=rescolors[2])

    if(isf==4):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo/yadism_sf_f3_nnlo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nlo/yadism_sf_f3_nnlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo/yadism_sf_f3_nnlo,ls="solid",color=rescolors[2])

    if(isf==5):
        p1=ax.plot(yadism_sf_x, yadism_sf_fl_lo/yadism_sf_fl_nnlo,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_fl_nlo/yadism_sf_fl_nnlo,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo/yadism_sf_fl_nnlo,ls="solid",color=rescolors[2])

    if(isf==6):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f2_nlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar,ls="solid",color=rescolors[2])

    if(isf==7):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar,ls="solid",color=rescolors[2])

    if(isf==8):
        p1=ax.plot(yadism_sf_x, yadism_sf_fl_lo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_fl_nlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo_nubar,ls="solid",color=rescolors[2])

    if(isf==9):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_lo_nubar/yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f2_nlo_nubar/yadism_sf_f2_nnlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar/yadism_sf_f2_nnlo_nubar,ls="solid",color=rescolors[2])

    if(isf==10):
        p1=ax.plot(yadism_sf_x, yadism_sf_f3_lo_nubar/yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nlo_nubar/yadism_sf_f3_nnlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar/yadism_sf_f3_nnlo_nubar,ls="solid",color=rescolors[2])

    if(isf==11):
        p1=ax.plot(yadism_sf_x, yadism_sf_fl_lo_nubar/yadism_sf_fl_nnlo_nubar,ls="dashed",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_fl_nlo_nubar/yadism_sf_fl_nnlo_nubar,ls="dashdot",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo_nubar/yadism_sf_fl_nnlo_nubar,ls="solid",color=rescolors[2])
       
    
    ax.set_xscale('log')
    ax.set_xlim(xmin,0.7)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>8):
        ax.set_xlabel(r'$x$',fontsize=15)
    if(isf==0):
        if(q > 9.9 and q < 10.1):
            ax.text(0.57,0.87,r'$Q=10~{\rm GeV}$',\
                    fontsize=16,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.62,0.87,r'$Q=2~{\rm GeV}$',\
                    fontsize=16,transform=ax.transAxes)
    if(isf==6):
        if(q > 9.9 and q < 10.1):
            ax.text(0.57,0.87,r'$Q=10~{\rm GeV}$',\
                    fontsize=16,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.62,0.87,r'$Q=2~{\rm GeV}$',\
                    fontsize=16,transform=ax.transAxes)

    if(isf==0):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"${\rm LO}$",r"${\rm NLO}$",r"${\rm NNLO}$"],
                  frameon="True",loc=3,prop={'size':14})
                        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-PerturbativeStab-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-PerturbativeStab-xdep'+filelabel+'.pdf')

#*******************************************************************************
#*******************************************************************************
#
# Study relative size of FL and xF3 when compared with
# the dominant structure function F2 at NNLO
#
print("\n ****** Plotting relative size of Structure Functions  ******* \n")

ncols,nrows=2,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if(q > 1.9 and q < 2.1):
    yranges=[[0,1.02],[0,1.02],[0,1.02],[0,1.02]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,1.02],[-0.2,1.02],[0,1.02],[-0.2,1.02]]

labelpdf=[r"$F_i^{\nu p} / F_2^{\nu p}$",r"$F_i^{\bar{\nu} p} / F_2^{\bar{\nu} p}$",\
          r"$F_i^{\nu p} / F_2^{\nu p}$",r"$F_i^{\bar{\nu} p} / F_2^{\bar{\nu} p}$"]

for isf in range(2):

    ax = py.subplot(gs[isf])

    # NNLO YADISM
    if(isf==0):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo/yadism_sf_f2_nnlo,\
                   ls="solid",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo/yadism_sf_f2_nnlo,\
                   ls="dashed",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo/yadism_sf_f2_nnlo,\
                   ls="dashdot",color=rescolors[2])

    # NNLO YADISM
    if(isf==1):
        p1=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar/yadism_sf_f2_nnlo_nubar,\
                   ls="solid",color=rescolors[0])
        p2=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar/yadism_sf_f2_nnlo_nubar,\
                   ls="dashed",color=rescolors[1])
        p3=ax.plot(yadism_sf_x, yadism_sf_fl_nnlo_nubar/yadism_sf_f2_nnlo_nubar,\
                   ls="dashdot",color=rescolors[2])
    
    
    ax.set_xscale('log')
    ax.set_xlim(xmin,0.7)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>-1):
        ax.set_xlabel(r'$x$',fontsize=16)
    if(isf==0):
        if(q > 9.9 and q < 10.1):
            ax.text(0.37,0.87,r'${\rm NNLO,\,} Q=10~{\rm GeV}$',\
                    fontsize=14,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.37,0.87,r'${\rm NNLO,\,} Q=2~{\rm GeV}$',\
                    fontsize=15,transform=ax.transAxes)
    if(isf==2):
        if(q > 9.9 and q < 10.1):
            ax.text(0.57,0.87,r'${\rm NNLO,\,} Q=10~{\rm GeV}$',\
                    fontsize=14,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.57,0.87,r'${\rm NNLO,\,} Q=2~{\rm GeV}$',\
                    fontsize=15,transform=ax.transAxes)

    if(isf==0):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"$F_2$",r"$xF_3$",r"$F_L$"],
                  frameon="True",loc=2,prop={'size':14})
                        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-RelSize-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-RelSize-xdep'+filelabel+'.pdf')

exit()

#*******************************************************************************
#*******************************************************************************

print("\n ****** Plotting absolute Structure Functions (Comparisons NNSF results) ******* \n")

ncols,nrows=2,2
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if(q > 1.9 and q < 2.1):
    yranges=[[0,2.3],[0,2.3],[0,0.9],[-0.25,1.40]]
if(q > 9.9 and q < 10.1):
    yranges=[[0,4.6],[0,4.6],[0,1.3],[-0.7,1.30]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    
    # GENIE BY
    if(isf==0):
        p1=ax.plot(genie_sf_x, genie_sf_f2,ls="solid",color=rescolors[3])
    if(isf==1):
        p1=ax.plot(genie_sf_x, genie_sf_f2_nubar,ls="solid",color=rescolors[3])
    if(isf==2):
        p1=ax.plot(genie_sf_x, genie_sf_f3,ls="solid",color=rescolors[3])
    if(isf==3):
        p1=ax.plot(genie_sf_x, genie_sf_f3_nubar,ls="solid",color=rescolors[3])

    # GENIE BGR18
    if(isf==0):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2,ls="dashdot",color=rescolors[5])
    if(isf==1):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f2_nub,ls="dashdot",color=rescolors[5])
    if(isf==2):
        p2=ax.plot(genie_sf_bgr_x, genie_sf_bgr_f3,ls="dashdot",color=rescolors[5])
    if(isf==3):
        p2=ax.plot(genie_sf_bgr_x, (-1)*genie_sf_bgr_f3_nub,ls="dashdot",color=rescolors[5])

    # NNLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo,ls="dashed",color=rescolors[4])
    if(isf==1):
        p3=ax.plot(yadism_sf_x, yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[4])
    if(isf==2):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo,ls="dashed",color=rescolors[4])
    if(isf==3):
        p3=ax.plot(yadism_sf_x, yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[4])

    ## NNSF machine learning parametrisation
    if(isf==0):
        p4=ax.plot(x_nnsf,nnsf_f2nu_mid,ls="dotted")
        ax.fill_between(x_nnsf,nnsf_f2nu_high,nnsf_f2nu_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==1):
        p4=ax.plot(x_nnsf,nnsf_f2nubar_mid,ls="dotted")
        ax.fill_between(x_nnsf,nnsf_f2nubar_high,nnsf_f2nubar_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==2):
        p4=ax.plot(x_nnsf,nnsf_xf3nu_mid,ls="dotted")
        ax.fill_between(x_nnsf,nnsf_xf3nu_high,nnsf_xf3nu_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==3):
        p4=ax.plot(x_nnsf,nnsf_xf3nubar_mid,ls="dotted")
        ax.fill_between(x_nnsf,nnsf_xf3nubar_high,nnsf_xf3nubar_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
      
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
        if(q > 9.9 and q < 10.1):
            ax.text(0.67,0.85,r'$Q=10~{\rm GeV}$',fontsize=14,transform=ax.transAxes)
        if(q > 1.9 and q < 2.1):
            ax.text(0.67,0.85,r'$Q=2~{\rm GeV}$',fontsize=15,transform=ax.transAxes)
             
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0],(p4[0],p5[0])],\
                  [r"${\rm Bodek\,Yang~(LO, GRV98)}$",\
                   r"${\rm BGR18~(NLO, NNPDF3.1)}$", \
                   r"${\rm YADISM~(NNLO, NNPDF4.0)}$",\
                   r"${\rm NNSF}$",], \
                  frameon="True",loc=3,prop={'size':10})
                
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Comparisons-xdep'+filelabel+'.pdf')
print('output plot: StructureFunction-Comparisons-xdep'+filelabel+'.pdf')

exit()


######################################################################
######################################################################

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
