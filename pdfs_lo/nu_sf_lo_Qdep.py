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
qmax=10
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
pdfsetlab=[r"{\sc LO-SF-NNPDF4.0}",r"{\sc LO-SF-GRV98}"]
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

#---------------------------
#---------------------------
print("\n Reading the YADISM structure functions \n")

# Read YADISM inputs

yadism_f2_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/F2.txt")
yadism_f3_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/F3.txt")
yadism_fl_lo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/LO/predictions/FL.txt")
yadism_f2_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/F2.txt")
yadism_f3_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/F3.txt")
yadism_fl_nlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NLO/predictions/FL.txt")
yadism_f2_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/F2.txt")
yadism_f3_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/F3.txt")
yadism_fl_nnlo_nu_p=np.loadtxt("Yadism_data_v2/neutrino/NNLO/predictions/FL.txt")

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
yadism_sf_q = np.zeros(nq2_yadism)
yadism_sf_f2_lo=np.zeros(nq2_yadism)
yadism_sf_f3_lo=np.zeros(nq2_yadism)
yadism_sf_fl_lo=np.zeros(nq2_yadism)
yadism_sf_f2_nlo=np.zeros(nq2_yadism)
yadism_sf_f3_nlo=np.zeros(nq2_yadism)
yadism_sf_fl_nlo=np.zeros(nq2_yadism)
yadism_sf_f2_nnlo=np.zeros(nq2_yadism)
yadism_sf_f3_nnlo=np.zeros(nq2_yadism)
yadism_sf_fl_nnlo=np.zeros(nq2_yadism)
yadism_sf_f2_lo_nubar=np.zeros(nq2_yadism)
yadism_sf_f3_lo_nubar=np.zeros(nq2_yadism)
yadism_sf_fl_lo_nubar=np.zeros(nq2_yadism)
yadism_sf_f2_nlo_nubar=np.zeros(nq2_yadism)
yadism_sf_f3_nlo_nubar=np.zeros(nq2_yadism)
yadism_sf_fl_nlo_nubar=np.zeros(nq2_yadism)
yadism_sf_f2_nnlo_nubar=np.zeros(nq2_yadism)
yadism_sf_f3_nnlo_nubar=np.zeros(nq2_yadism)
yadism_sf_fl_nnlo_nubar=np.zeros(nq2_yadism)


if(x > 0.0125 and x < 0.0127):
    ix_yadism=10
if(x > 0.24 and x < 0.26):
    ix_yadism=23  

# Checkx
print(yadism_f2_lo_nu_p)
x_check=yadism_f2_lo_nu_p[nq2_yadism*ix_yadism][1]
reldiff=abs( (x_check-x) /x )
print(x_check," ",x)
if(reldiff > 0.04):
    print("x mismatch")
    print(x_check," ",x, " ",reldiff)
    exit()

icount=0
for iq2 in range(nq2_yadism):
    index = nq2_yadism*ix_yadism + iq2
    print(iq2," ",index," ",yadism_f2_lo_nu_p[index][1]," ",math.pow(yadism_f2_lo_nu_p[index][2],0.5))
    # Neutrino SFs
    yadism_sf_q[iq2] = math.pow(yadism_f2_lo_nu_p[index][2],0.5)
    yadism_sf_f2_lo[iq2] = yadism_f2_lo_nu_p[index][3]
    yadism_sf_f3_lo[iq2] = yadism_f3_lo_nu_p[index][3]
    yadism_sf_fl_lo[iq2] = yadism_fl_lo_nu_p[index][3]
    yadism_sf_f2_nlo[iq2] = yadism_f2_nlo_nu_p[index][3]
    yadism_sf_f3_nlo[iq2] = yadism_f3_nlo_nu_p[index][3]
    yadism_sf_fl_nlo[iq2] = yadism_fl_nlo_nu_p[index][3]
    yadism_sf_f2_nnlo[iq2] = yadism_f2_nnlo_nu_p[index][3]
    yadism_sf_f3_nnlo[iq2] = yadism_f3_nnlo_nu_p[index][3]
    yadism_sf_fl_nnlo[iq2] = yadism_fl_nnlo_nu_p[index][3]
    # Anti-Neutrino SFs
    yadism_sf_f2_lo_nubar[iq2] = yadism_f2_lo_nubar_p[index][3]
    yadism_sf_f3_lo_nubar[iq2] = yadism_f3_lo_nubar_p[index][3]
    yadism_sf_fl_lo_nubar[iq2] = yadism_fl_lo_nubar_p[index][3]
    yadism_sf_f2_nlo_nubar[iq2] = yadism_f2_nlo_nubar_p[index][3]
    yadism_sf_f3_nlo_nubar[iq2] = yadism_f3_nlo_nubar_p[index][3]
    yadism_sf_fl_nlo_nubar[iq2] = yadism_fl_nlo_nubar_p[index][3]
    yadism_sf_f2_nnlo_nubar[iq2] = yadism_f2_nnlo_nubar_p[index][3]
    yadism_sf_f3_nnlo_nubar[iq2] = yadism_f3_nnlo_nubar_p[index][3]
    yadism_sf_fl_nnlo_nubar[iq2] = yadism_fl_nnlo_nubar_p[index][3] 
    icount = icount+1



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

ncols,nrows=4,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if( x > 0.0125 and x < 0.0127):
    yranges=[[1.3,3.3],[1.3,3.3],[0.25,0.7],[-0.15,0.25]]
if( x > 0.24 and x < 0.26):
    yranges=[[0.53,0.80],[1.07,1.5],[0.43,0.68],[0.9,1.33]]
    

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    # LO SF + NNPDF4.0
    p1=ax.plot(Q,p1_mid[isf],ls="dotted")
    ax.fill_between(Q,p1_high[isf],p1_low[isf],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    
    # LO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_lo,ls="dashed",color=rescolors[1])
    if(isf==1):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_lo_nubar,ls="dashed",color=rescolors[1])
    if(isf==2):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_lo,ls="dashed",color=rescolors[1])
    if(isf==3):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_lo_nubar,ls="dashed",color=rescolors[1])
        
    # LO SF + GRV98
    p4=ax.plot(Q,p2_mid[isf],ls="dotted",color=rescolors[2],lw=3)

    # GENIE BY
    if(isf==0):
        p5=ax.plot(genie_sf_q, genie_sf_f2,ls="solid",color=rescolors[3])
    if(isf==1):
        p5=ax.plot(genie_sf_q, genie_sf_f2_nubar,ls="solid",color=rescolors[3])
    if(isf==2):
        p5=ax.plot(genie_sf_q, genie_sf_f3,ls="solid",color=rescolors[3])
    if(isf==3):
        p5=ax.plot(genie_sf_q, genie_sf_f3_nubar,ls="solid",color=rescolors[3])
        
    ax.set_xscale('linear')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>-1):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=15)
    if( x > 0.0125 and x < 0.0127):
        if(isf==0):
            ax.text(0.05,0.88,r'$x=0.0126$',fontsize=17,transform=ax.transAxes)
    if( x > 0.24 and x < 0.26):
        if(isf==0):
            ax.text(0.70,0.88,r'$x=0.25$',fontsize=17,transform=ax.transAxes)
    
    if(isf==1):
        ax.legend([(p1[0],p2[0]),p3[0],p4[0],p5[0]],\
                  [pdfsetlab[0],r"{\sc YADISM-LO}",\
                   pdfsetlab[1],\
                   r"{\sc Bodek-Yang}"], \
                  frameon="True",loc=4,prop={'size':12})
        

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Validation'+filelabel+'.pdf')
print('output plot: StructureFunction-Validation'+filelabel+'.pdf')

#################################################################################
#################################################################################


print("\n ****** Plotting absolute Structure Functions (Perturbative Convergence) ******* \n")

py.clf()
ncols,nrows=2,2
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if( x > 0.0125 and x < 0.0127):
    yranges=[[1.3,3.3],[1.3,3.3],[0.25,0.7],[-0.15,0.25]]
if( x > 0.24 and x < 0.26):
    yranges=[[0.5,0.80],[1.05,1.5],[0.38,0.65],[0.9,1.25]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])

    # LO YADISM
    if(isf==0):
        p1=ax.plot(yadism_sf_q, yadism_sf_f2_lo,ls="dashed",color=rescolors[0])
    if(isf==1):
        p1=ax.plot(yadism_sf_q, yadism_sf_f2_lo_nubar,ls="dashed",color=rescolors[0])
    if(isf==2):
        p1=ax.plot(yadism_sf_q, yadism_sf_f3_lo,ls="dashed",color=rescolors[0])
    if(isf==3):
        p1=ax.plot(yadism_sf_q, yadism_sf_f3_lo_nubar,ls="dashed",color=rescolors[0])

    # GENIE BGR18 (NNPDF3.1 NLO)
    if(isf==0):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2,ls="dashdot",color=rescolors[1])
    if(isf==1):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2_nubar,ls="dashdot",color=rescolors[1])
    if(isf==2):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3,ls="dashdot",color=rescolors[1])
    if(isf==3):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3_nubar,ls="dashdot",color=rescolors[1])

    # NNLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nnlo,ls="solid",color=rescolors[2])
    if(isf==1):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nnlo_nubar,ls="solid",color=rescolors[2])
    if(isf==2):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nnlo,ls="solid",color=rescolors[2])
    if(isf==3):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nnlo_nubar,ls="solid",color=rescolors[2])
        
    ax.set_xscale('linear')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>1):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=15)

    if( x > 0.0125 and x < 0.0127):
        if(isf==0):
            ax.text(0.05,0.87,r'$x=0.0126$',fontsize=16,transform=ax.transAxes)
    if( x > 0.24 and x < 0.26):
        if(isf==0):
            ax.text(0.65,0.85,r'$x=0.25$',fontsize=17,transform=ax.transAxes)
    
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"${\rm YADISM~(LO)+NNPDF4.0}$",\
                   r"${\rm BGR18 NLO (NNPDF3.1)~{(GENIE)}}$",\
                   r"${\rm YADISM~(NNLO)+NNPDF4.0}$"],
                  frameon="True",loc=1,prop={'size':11})

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Perturbative'+filelabel+'.pdf')
print('output plot: StructureFunction-Perturbative'+filelabel+'.pdf')

#################################################################################
#################################################################################

print("\n ****** Plotting absolute Structure Functions (Comparisons pre-fit) ******* \n")

py.clf()
ncols,nrows=4,1
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if( x > 0.0125 and x < 0.0127):
    yranges=[[1.2,3.6],[1.2,3.6],[0.2,0.7],[-0.15,0.30]]
if( x > 0.24 and x < 0.26):
    yranges=[[0.5,0.8],[1.0,1.4],[0.35,0.55],[0.9,1.12]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])
        
    # GENIE BY
    if(isf==0):
        p1=ax.plot(genie_sf_q, genie_sf_f2,ls="solid",color=rescolors[3],lw=2)
    if(isf==1):
        p1=ax.plot(genie_sf_q, genie_sf_f2_nubar,ls="solid",color=rescolors[3],lw=2)
    if(isf==2):
        p1=ax.plot(genie_sf_q, genie_sf_f3,ls="solid",color=rescolors[3],lw=2)
    if(isf==3):
        p1=ax.plot(genie_sf_q, genie_sf_f3_nubar,ls="solid",color=rescolors[3],lw=2)

     # GENIE BGR18
    if(isf==0):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==1):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2_nubar,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==2):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==3):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3_nubar,ls="dashdot",color=rescolors[5],lw=2)

    # NNLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nlo,ls="dashed",color=rescolors[4],lw=2)
    if(isf==1):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nlo_nubar,ls="dashed",color=rescolors[4],lw=2)
    if(isf==2):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nlo,ls="dashed",color=rescolors[4],lw=2)
    if(isf==3):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nlo_nubar,ls="dashed",color=rescolors[4],lw=2)

    ax.set_xscale('linear')
    ax.set_xlim(qmin,qmax)
    ax.tick_params(which='both',direction='in',labelsize=13,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=16)

    if( x > 0.0125 and x < 0.0127):
        if(isf==0):
            ax.text(0.05,0.87,r'$x=0.0126$',fontsize=18,transform=ax.transAxes)
    if( x > 0.24 and x < 0.26):
        if(isf==0):
            ax.text(0.65,0.85,r'$x=0.25$',fontsize=18,transform=ax.transAxes)
    
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0]],\
                  [r"{\sc Bodek-Yang}",\
                   r"{\sc BGR18}",\
                   r"{\sc YADISM-NLO}"],
                  frameon="True",loc=4,prop={'size':12})

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-ComparisonsPreFit'+filelabel+'.pdf')
print('output plot: StructureFunction-ComparisonsPreFit'+filelabel+'.pdf')

################################################################################
#################################################################################

print("\n ****** Plotting absolute Structure Functions (Comparisons) ******* \n")

py.clf()
ncols,nrows=2,2
py.figure(figsize=(ncols*5,nrows*3.5))
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

if( x > 0.0125 and x < 0.0127):
    yranges=[[0.3,3.1],[0.3,3.1],[0.0,0.7],[-0.15,0.60]]
if( x > 0.24 and x < 0.26):
    yranges=[[0.30,1.2],[0.9,2.0],[0.2,0.9],[0.3,1.5]]

labelpdf=[r"$F_2^{\nu p}(x,Q)$",r"$F_2^{\bar{\nu} p}(x,Q)$",\
          r"$xF_3^{\nu p}(x,Q)$",r"$xF_3^{\bar{\nu} p}(x,Q)$"]

for isf in range(nsf):

    ax = py.subplot(gs[isf])
        
    # GENIE BY
    if(isf==0):
        p1=ax.plot(genie_sf_q, genie_sf_f2,ls="solid",color=rescolors[3],lw=2)
    if(isf==1):
        p1=ax.plot(genie_sf_q, genie_sf_f2_nubar,ls="solid",color=rescolors[3],lw=2)
    if(isf==2):
        p1=ax.plot(genie_sf_q, genie_sf_f3,ls="solid",color=rescolors[3],lw=2)
    if(isf==3):
        p1=ax.plot(genie_sf_q, genie_sf_f3_nubar,ls="solid",color=rescolors[3],lw=2)

     # GENIE BGR18
    if(isf==0):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==1):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f2_nubar,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==2):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3,ls="dashdot",color=rescolors[5],lw=2)
    if(isf==3):
        p2=ax.plot(genie_bgr_sf_q, genie_bgr_sf_f3_nubar,ls="dashdot",color=rescolors[5],lw=2)

    # NLO YADISM
    if(isf==0):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nnlo,ls="dashed",color=rescolors[4],lw=2)
    if(isf==1):
        p3=ax.plot(yadism_sf_q, yadism_sf_f2_nnlo_nubar,ls="dashed",color=rescolors[4],lw=2)
    if(isf==2):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nnlo,ls="dashed",color=rescolors[4],lw=2)
    if(isf==3):
        p3=ax.plot(yadism_sf_q, yadism_sf_f3_nnlo_nubar,ls="dashed",color=rescolors[4],lw=2)

    ## NNSF machine learning parametrisation
    if(isf==0):
        p4=ax.plot(q_nnsf,nnsf_f2nu_mid,ls="dotted")
        ax.fill_between(q_nnsf,nnsf_f2nu_high,nnsf_f2nu_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==1):
        p4=ax.plot(q_nnsf,nnsf_f2nubar_mid,ls="dotted")
        ax.fill_between(q_nnsf,nnsf_f2nubar_high,nnsf_f2nubar_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==2):
        p4=ax.plot(q_nnsf,nnsf_xf3nu_mid,ls="dotted")
        ax.fill_between(q_nnsf,nnsf_xf3nu_high,nnsf_xf3nu_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    if(isf==3):
        p4=ax.plot(q_nnsf,nnsf_xf3nubar_mid,ls="dotted")
        ax.fill_between(q_nnsf,nnsf_xf3nubar_high,nnsf_xf3nubar_low,color=rescolors[0],alpha=0.2)
        p5=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    
    #ax.set_xscale('linear')
    ax.set_xscale('log')
    ax.set_xlim(0.3,qmax)
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_xticks([0.3, 0.7, 1.0, 2.0, 4.0, 10.0])
    ax.set_ylabel(labelpdf[isf],fontsize=17)
    ax.set_ylim(yranges[isf][0],yranges[isf][1])
    if(isf>1):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=15)

    if( x > 0.0125 and x < 0.0127):
        if(isf==0):
            ax.text(0.05,0.87,r'$x=0.0126$',fontsize=16,transform=ax.transAxes)
    if( x > 0.24 and x < 0.26):
        if(isf==0):
            ax.text(0.65,0.85,r'$x=0.25$',fontsize=17,transform=ax.transAxes)
    
    if(isf==1):
        ax.legend([p1[0],p2[0],p3[0],(p4[0],p5[0])],\
                  [r"${\rm Bodek\,Yang~(LO, GRV98)}$",\
                   r"${\rm BGR18~(NLO, NNPDF3.1)}$", \
                   r"${\rm YADISM~(NNLO, NNPDF4.0)}$",\
                   r"${\rm NNSF}$",], \
                  frameon="True",loc=2,prop={'size':10})


py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Comparisons'+filelabel+'.pdf')
print('output plot: StructureFunction-Comparisons'+filelabel+'.pdf')

exit()

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


isf=0
ax = py.subplot(gs[isf])

# F2nu
p1=ax.plot(yadism_sf_q,  yadism_nnlo_f2/yadism_f2,ls="solid",lw=2,color=rescolors[0])

# F2nubar
p2=ax.plot(yadism_sf_q,  yadism_nnlo_f2_nubar/yadism_f2_nubar,ls="dashed",lw=2,color=rescolors[1])

# xF3nu
p3=ax.plot(yadism_sf_q,  yadism_nnlo_f3/yadism_f3,ls="solid",lw=2,color=rescolors[2])

# xF3nubar
p4=ax.plot(yadism_sf_q,  yadism_nnlo_f3_nubar/yadism_f3_nubar,ls="dashed",lw=2,color=rescolors[3])


ax.set_xscale('linear')
ax.set_xlim(qmin,qmax)
ax.tick_params(which='both',direction='in',labelsize=12,right=True)
ax.tick_params(which='major',length=7)
ax.tick_params(which='minor',length=4)
ax.set_ylabel(r"${\rm NNLO/LO~}K\,{\rm factor}$",fontsize=17)
# x = 0.25
ax.set_ylim(0.80,1.01)
ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=15)
#ax.text(0.08,0.87,r'$x=0.25$',fontsize=17,transform=ax.transAxes)
ax.text(0.08,0.87,r'$x=0.0126$',fontsize=17,transform=ax.transAxes)

ax.legend([p1[0],p2[0],p3[0],p4[0]],[r"$F_2^{\nu p}$",r"$F_2^{\bar{\nu} p}$",r"$xF_3^{\nu p}$",r"$xF_3^{\bar{\nu} p}$"], \
          frameon="True",loc=4,prop={'size':14})
        
py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('StructureFunction-Qdep-Kfact'+filelabel+'.pdf')
print('output plot: StructureFunction-Qdep-Kfact'+filelabel+'.pdf')




exit()
