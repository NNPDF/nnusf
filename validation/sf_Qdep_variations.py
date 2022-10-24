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

print("\n *********************************************************")
print("      Compute Q dependence of NNUSF                         ")
print(" ***********************************************************\n")

nset=6

# Comparison between baseline and fits without matching
#pdfset=["NNUSF10_A1_Q2MIN001_FIXQ2","NNUSF10_A1_Q2MIN001_NOMATCHING",\
#        "NNUSF10_A56_Q2MIN001_FIXQ2","NNUSF10_A56_Q2MIN001_NOMATCHING",\
#        "NNUSF10_A208_Q2MIN001_FIXQ2","NNUSF10_A208_Q2MIN001_NOMATCHING"]
#desc="nomatching"

# Comparison between baseline and fits neglecting nuclear variations
pdfset=["NNUSF10_A1_Q2MIN001_FIXQ2","NNUSF10_A1_Q2MIN001_TREATALL_ASPROTON",\
        "NNUSF10_A56_Q2MIN001_FIXQ2","NNUSF10_A1_Q2MIN001_TREATALL_ASPROTON",\
        "NNUSF10_A208_Q2MIN001_FIXQ2","NNUSF10_A1_Q2MIN001_TREATALL_ASPROTON"]
desc="nonuclear"


#---------------------------------------------------------
#---------------------------------------------------------
# Choice of x value
#filelabel = "x0p01"
filelabel = "x0p1"
#filelabel = "x0p00126"

# Plot labels
if(filelabel == "x0p00126"):
    x = 0.00126
if(filelabel=="x0p01"):
    x = 1e-2
if(filelabel=="x0p1"):
    x = 0.1 

#---------------------------------------------------------
#---------------------------------------------------------
# Read the YADISM NNLO proton predictions

# Neutrino
yadism_f2_nnlo_nu_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/neutrino/NNLO/predictions/F2.txt")
yadism_xf3_nnlo_nu_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/neutrino/NNLO/predictions/F3.txt")
yadism_fl_nnlo_nu_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/neutrino/NNLO/predictions/FL.txt")

# Anti-neutrino
yadism_f2_nnlo_nubar_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/antineutrino/NNLO/predictions/F2.txt")
yadism_xf3_nnlo_nubar_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/antineutrino/NNLO/predictions/F3.txt")
yadism_fl_nnlo_nubar_p=np.loadtxt("../pdfs_lo/Yadism_data_v2/antineutrino/NNLO/predictions/FL.txt")

print

nq_yadism=20
nx_yadism =30
nrep_yadism=100

yadism_sf_q=np.zeros(nq_yadism)
yadism_nnlo_f2_nu=np.zeros((nrep_yadism,nq_yadism))
yadism_nnlo_xf3_nu=np.zeros((nrep_yadism,nq_yadism))
yadism_nnlo_fl_nu=np.zeros((nrep_yadism,nq_yadism))
yadism_nnlo_f2_nubar=np.zeros((nrep_yadism,nq_yadism))
yadism_nnlo_xf3_nubar=np.zeros((nrep_yadism,nq_yadism))
yadism_nnlo_fl_nubar=np.zeros((nrep_yadism,nq_yadism))

ix_yadism = 1e3
if(x > 0.001 and x < 0.0015):
    ix_yadism=0  # x = 0.00126
if(x > 0.009 and x < 0.012):
    ix_yadism=9  # x = 0.01
if(x > 0.099 and x < 0.11):
    ix_yadism=19  # x = 0.1

icount=0
for irep in range(0,nrep_yadism+1):
    for ix in range(nx_yadism):
        for iq in range(nq_yadism):
            x_tmp = yadism_f2_nnlo_nu_p[icount][1]
            q_tmp = math.sqrt(yadism_f2_nnlo_nu_p[icount][2])
            #print(irep," ",ix," ",iq," ",x_tmp," ",q_tmp, " ",icount)
            if(ix==0):
                yadism_sf_q[iq] = q_tmp
            if(ix==ix_yadism):
                # First check
                reldiff=abs( (x_tmp-x) /x )
                if(reldiff > 1e-3):
                    print("x mismatch")
                    print(x_tmp," ",x," ",reldiff)
                    exit()
                # Now fill
                if(irep > 0):
                    print(yadism_f2_nnlo_nu_p[icount][3])
                    yadism_nnlo_f2_nu[irep-1][iq] = yadism_f2_nnlo_nu_p[icount][3]
                    yadism_nnlo_xf3_nu[irep-1][iq] = yadism_xf3_nnlo_nu_p[icount][3]
                    yadism_nnlo_fl_nu[irep-1][iq] = yadism_fl_nnlo_nu_p[icount][3]
                    yadism_nnlo_f2_nubar[irep-1][iq] = yadism_f2_nnlo_nubar_p[icount][3]
                    yadism_nnlo_xf3_nubar[irep-1][iq] = yadism_xf3_nnlo_nubar_p[icount][3]
                    yadism_nnlo_fl_nubar[irep-1][iq] = yadism_fl_nnlo_nubar_p[icount][3]
            
            icount = icount+1

#print(yadism_sf_q)
#print(yadism_nnlo_f2_nu)
#print(yadism_nnlo_f2_nu)

yadism_nnlo_f2_nu_high = np.nanpercentile(yadism_nnlo_f2_nu,84,axis=0)
yadism_nnlo_f2_nu_low = np.nanpercentile(yadism_nnlo_f2_nu,16,axis=0)
yadism_nnlo_f2_nu_mid = ( yadism_nnlo_f2_nu_high + yadism_nnlo_f2_nu_low )/2.
yadism_nnlo_f2_nu_error = ( yadism_nnlo_f2_nu_high - yadism_nnlo_f2_nu_low )/2.

print(yadism_nnlo_f2_nu_high)
print(yadism_nnlo_f2_nu_mid)
print(yadism_nnlo_f2_nu_low)

yadism_nnlo_f2_nubar_high = np.nanpercentile(yadism_nnlo_f2_nubar,84,axis=0)
yadism_nnlo_f2_nubar_low = np.nanpercentile(yadism_nnlo_f2_nubar,16,axis=0)
yadism_nnlo_f2_nubar_mid = ( yadism_nnlo_f2_nubar_high + yadism_nnlo_f2_nubar_low )/2.
yadism_nnlo_f2_nubar_error = ( yadism_nnlo_f2_nubar_high - yadism_nnlo_f2_nubar_low )/2.

yadism_nnlo_fl_nu_high = np.nanpercentile(yadism_nnlo_fl_nu,84,axis=0)
yadism_nnlo_fl_nu_low = np.nanpercentile(yadism_nnlo_fl_nu,16,axis=0)
yadism_nnlo_fl_nu_mid = ( yadism_nnlo_fl_nu_high + yadism_nnlo_fl_nu_low )/2.
yadism_nnlo_fl_nu_error = ( yadism_nnlo_fl_nu_high - yadism_nnlo_fl_nu_low )/2.

yadism_nnlo_fl_nubar_high = np.nanpercentile(yadism_nnlo_fl_nubar,84,axis=0)
yadism_nnlo_fl_nubar_low = np.nanpercentile(yadism_nnlo_fl_nubar,16,axis=0)
yadism_nnlo_fl_nubar_mid = ( yadism_nnlo_fl_nubar_high + yadism_nnlo_fl_nubar_low )/2.
yadism_nnlo_fl_nubar_error = ( yadism_nnlo_fl_nubar_high - yadism_nnlo_fl_nubar_low )/2.

yadism_nnlo_xf3_nu_high = np.nanpercentile(yadism_nnlo_xf3_nu,84,axis=0)
yadism_nnlo_xf3_nu_low = np.nanpercentile(yadism_nnlo_xf3_nu,16,axis=0)
yadism_nnlo_xf3_nu_mid = ( yadism_nnlo_xf3_nu_high + yadism_nnlo_xf3_nu_low )/2.
yadism_nnlo_xf3_nu_error = ( yadism_nnlo_xf3_nu_high - yadism_nnlo_xf3_nu_low )/2.

yadism_nnlo_xf3_nubar_high = np.nanpercentile(yadism_nnlo_xf3_nubar,84,axis=0)
yadism_nnlo_xf3_nubar_low = np.nanpercentile(yadism_nnlo_xf3_nubar,16,axis=0)
yadism_nnlo_xf3_nubar_mid = ( yadism_nnlo_xf3_nubar_high + yadism_nnlo_xf3_nubar_low )/2.
yadism_nnlo_xf3_nubar_error = ( yadism_nnlo_xf3_nubar_high - yadism_nnlo_xf3_nubar_low )/2.



print("\n *********************************************************")
print("      YADISM calculations read and processed                         ")
print(" ***********************************************************\n")


#---------------------------------------------------------
#---------------------------------------------------------
# General plot settings

# Plot labels
if(filelabel=="x0p1"):
    stringx=r'$x=0.1$'
if(filelabel=="x0p01"):
    stringx=r'$x=10^{-2}$'
if(filelabel=="x0p00126"):
    stringx=r'$x=0.00126$'
    
nq = 200
qmin = pow(1e-3,0.5)
qmax = 1000 

# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax=1000
# number of flavours to be plotted
nfl=9
# Set x grid
qgrid = np.logspace(log(qmin),log(qmax),nq)

nrep=np.zeros(nset, dtype='int')
nrep_max = 140

fit1 = np.zeros((nrep_max,nfl,nq))
fit2 = np.zeros((nrep_max,nfl,nq))
fit3 = np.zeros((nrep_max,nfl,nq))
fit4 = np.zeros((nrep_max,nfl,nq))
fit5 = np.zeros((nrep_max,nfl,nq))
fit6 = np.zeros((nrep_max,nfl,nq))

ids = np.array([1001,1002,1003,2001,2002,2003,3001,3002,3003])

for iset in range(nset):

    p=lhapdf.getPDFSet(pdfset[iset])
    nrep[iset]=int(p.get_entry("NumMembers"))-1
    print("nrep = ",nrep[iset])
    if(nrep[iset] > nrep_max):
        nrep[iset] = nrep_max
    if(nrep[iset] > nrep_max):
        print("Problem, too many replicas \n")
        print(nrep[iset]," ",nrep_max)
        exit()
    #print(p.description)
    print("nrep (updated) = ",nrep[iset])
    
    # Run over replicas
    for i in range(1,nrep[iset]+1):
        p=lhapdf.mkPDF(pdfset[iset],i)
        lhapdf.setVerbosity(0)
        
        # Run over x arrat
        for k in range(nq):
            
            q = qgrid[k]
            
            # run over flavours
            for ifl in range(nfl):

                #print(x," ",q)
                if(iset==0):
                    fit1[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==1):
                    fit2[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==2):
                    fit3[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==3):
                    fit4[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==4):
                    fit5[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                if(iset==5):
                    fit6[i-1][ifl][k] = p.xfxQ(ids[ifl],x,q)
                
                
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

p4_high = np.nanpercentile(fit4,84,axis=0)
p4_low = np.nanpercentile(fit4,16,axis=0)
p4_mid = ( p4_high + p4_low )/2.
p4_error = ( p4_high - p4_low )/2.

p5_high = np.nanpercentile(fit5,84,axis=0)
p5_low = np.nanpercentile(fit5,16,axis=0)
p5_mid = ( p5_high + p5_low )/2.
p5_error = ( p5_high - p5_low )/2.

p6_high = np.nanpercentile(fit6,84,axis=0)
p6_low = np.nanpercentile(fit6,16,axis=0)
p6_mid = ( p6_high + p6_low )/2.
p6_error = ( p6_high - p6_low )/2.


#---------------------------------------------------------------------
# Plot absolute SFs
#---------------------------------------------------------------------

py.clf()
ncols,nrows=3,6
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
for ifl in range(6):

    ax = py.subplot(gs[icount])
    
    ## NNUSF proton
    p1=ax.plot(qgrid,p1_mid[ifl],ls="solid")
    ax.fill_between(qgrid,p1_high[ifl],p1_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    ## NNUSF proton (variation)
    p3=ax.plot(qgrid,p2_mid[ifl],ls="dashed")
    ax.fill_between(qgrid,p2_high[ifl],p2_low[ifl],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)

    ## YADISM NNPDF4.0 NNLO proton baseline
    if(ifl==0):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_f2_nu_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_f2_nu_high,yadism_nnlo_f2_nu_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
    if(ifl==1):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_fl_nu_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_fl_nu_high,yadism_nnlo_fl_nu_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
    if(ifl==2):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_xf3_nu_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_xf3_nu_high,yadism_nnlo_xf3_nu_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)

    if(ifl==3):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_f2_nubar_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_f2_nubar_high,yadism_nnlo_f2_nubar_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
    if(ifl==4):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_fl_nubar_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_fl_nubar_high,yadism_nnlo_fl_nubar_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
    if(ifl==5):
        p7 = ax.plot(yadism_sf_q,yadism_nnlo_xf3_nubar_mid,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,yadism_nnlo_xf3_nubar_high,yadism_nnlo_xf3_nubar_low,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)

    if(ifl==6):
        p7 = ax.plot(yadism_sf_q,(yadism_nnlo_f2_nu_mid+yadism_nnlo_f2_nubar_mid)/2,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,(yadism_nnlo_f2_nu_high+yadism_nnlo_f2_nubar_high)/2,(yadism_nnlo_f2_nu_low+yadism_nnlo_f2_nubar_low)/2,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)

    if(ifl==7):
        p7 = ax.plot(yadism_sf_q,(yadism_nnlo_fl_nu_mid+yadism_nnlo_fl_nubar_mid)/2,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,(yadism_nnlo_fl_nu_high+yadism_nnlo_fl_nubar_high)/2,\
                        (yadism_nnlo_fl_nu_low+yadism_nnlo_fl_nubar_low)/2,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)

    if(ifl==8):
        p7 = ax.plot(yadism_sf_q,(yadism_nnlo_xf3_nu_mid+yadism_nnlo_xf3_nubar_mid)/2,ls="dashdot",color=rescolors[3],lw=2)
        ax.fill_between(yadism_sf_q,(yadism_nnlo_xf3_nu_high+yadism_nnlo_xf3_nubar_high)/2,\
                        (yadism_nnlo_xf3_nu_low+yadism_nnlo_xf3_nubar_low)/2,color=rescolors[3],alpha=0.2)
        p8=ax.fill(np.NaN,np.NaN,color=rescolors[3],alpha=0.2)
            

    ax.set_xscale('log')
    ax.set_xlim(qmin,qmax)

    if(filelabel=="x0p1"):
        if(ifl==0): ax.set_ylim(0.3,2.2)
        if(ifl==3): ax.set_ylim(0.3,2.2)
        if(ifl==1): ax.set_ylim(-0.5,0.6)
        if(ifl==4): ax.set_ylim(-0.5,0.6)
        if(ifl==2): ax.set_ylim(-0.1,1.4)
        if(ifl==5): ax.set_ylim(-0.1,1.4)
                
    if(filelabel=="x0p01"):
        if(ifl==0): ax.set_ylim(-0.1,4.0)
        if(ifl==3): ax.set_ylim(-0.1,4.0)
        if(ifl==1): ax.set_ylim(-0.2,1.4)
        if(ifl==4): ax.set_ylim(-0.2,1.4)
        if(ifl==2): ax.set_ylim(-0.2,1.5)
        if(ifl==5): ax.set_ylim(-0.2,1.5)
        
    if(filelabel=="x0p00126"):
        if(ifl==0): ax.set_ylim(-0.1,7.0)
        if(ifl==3): ax.set_ylim(-0.1,7.0)
        if(ifl==1): ax.set_ylim(-0.2,1.8)
        if(ifl==4): ax.set_ylim(-0.2,1.8)
        if(ifl==2): ax.set_ylim(-0.2,1.7)
        if(ifl==5): ax.set_ylim(-0.7,1.7)
        
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=16)
    
    
    if(ifl==0):
        ax.text(0.10,0.85,stringx,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p7[0],p8[0]),(p3[0],p4[0])],\
                  #[ r"$A=1$", r"$A=1~{\rm (pQCD)}$",r"$A=1~(\rm wo~match)$" ], \
                  [ r"$A=1$", r"$A=1~{\rm (pQCD)}$",r"${\rm no~A~dep~in~NN}$" ], \
                  frameon=True,loc=4,prop={'size':13})

    # Add line indicating yadism region
    plt.axvline(5,lw=2,ls="dotted",color="black")
    if(ifl==2):
        ax.text(0.55,0.10,r'$\rm pQCD~constraints$',fontsize=14,transform=ax.transAxes,color="black")
        ax.arrow(x=0.55, y=0.05, dx=0.30, dy=0, width=.02,head_length=0.05,transform=ax.transAxes)

    icount = icount + 1

for ifl in range(6):

    ax = py.subplot(gs[icount])
    
    ## NNUSF iron
    p1=ax.plot(qgrid,p3_mid[ifl],ls="solid")
    ax.fill_between(qgrid,p3_high[ifl],p3_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    ## NNUSF iron
    p3=ax.plot(qgrid,p4_mid[ifl],ls="dashed")
    ax.fill_between(qgrid,p4_high[ifl],p4_low[ifl],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlim(qmin,qmax)

    if(filelabel=="x0p1"):
        if(ifl==0): ax.set_ylim(0.5,2.2)
        if(ifl==3): ax.set_ylim(0.5,2.2)
        if(ifl==1): ax.set_ylim(-0.5,0.6)
        if(ifl==4): ax.set_ylim(-0.5,0.6)
        if(ifl==2): ax.set_ylim(-0.1,1.4)
        if(ifl==5): ax.set_ylim(-0.1,1.4)

    if(filelabel=="x0p01"):
        if(ifl==0): ax.set_ylim(-0.1,4.5)
        if(ifl==3): ax.set_ylim(-0.1,4.5)
        if(ifl==1): ax.set_ylim(-0.7,1.4)
        if(ifl==4): ax.set_ylim(-0.7,1.4)
        if(ifl==2): ax.set_ylim(-0.4,1.6)
        if(ifl==5): ax.set_ylim(-0.4,1.6)
        
    if(filelabel=="x0p00126"):
        if(ifl==0): ax.set_ylim(-0.1,7.0)
        if(ifl==3): ax.set_ylim(-0.1,7.0)
        if(ifl==1): ax.set_ylim(-0.2,1.8)
        if(ifl==4): ax.set_ylim(-0.2,1.8)
        if(ifl==2): ax.set_ylim(-0.2,1.7)
        if(ifl==5): ax.set_ylim(-0.7,1.7)
        
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=16)
    
    if(ifl==0):
        ax.text(0.10,0.85,stringx,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0])],\
                  # [ r"$A=56$", r"$A=56~(\rm wo~match)$" ], \
                  [ r"$A=56$", r"${\rm no~A~dep~in~NN}$" ], \
                  frameon=True,loc=4,prop={'size':13})

    icount = icount + 1

    # Add line indicating yadism region
    plt.axvline(5,lw=2,ls="dotted",color="black")
    if(ifl==2):
        ax.text(0.55,0.10,r'$\rm pQCD~constraints$',fontsize=14,transform=ax.transAxes,color="black")
        ax.arrow(x=0.55, y=0.05, dx=0.30, dy=0, width=.02,head_length=0.05,transform=ax.transAxes)
        

for ifl in range(6):

    ax = py.subplot(gs[icount])
    
    ## NNUSF lead
    p1=ax.plot(qgrid,p5_mid[ifl],ls="solid")
    ax.fill_between(qgrid,p5_high[ifl],p5_low[ifl],color=rescolors[0],alpha=0.2)
    p2=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)

    ## NNUSF lead (variation)
    p3=ax.plot(qgrid,p6_mid[ifl],ls="dashed")
    ax.fill_between(qgrid,p6_high[ifl],p6_low[ifl],color=rescolors[1],alpha=0.2)
    p4=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlim(qmin,qmax)

    if(filelabel=="x0p1"):
        if(ifl==0): ax.set_ylim(0.5,2.2)
        if(ifl==3): ax.set_ylim(0.5,2.2)
        if(ifl==1): ax.set_ylim(-0.5,0.6)
        if(ifl==4): ax.set_ylim(-0.5,0.6)
        if(ifl==2): ax.set_ylim(-0.1,1.4)
        if(ifl==5): ax.set_ylim(-0.1,1.4)
        
    if(filelabel=="x0p01"):
        if(ifl==0): ax.set_ylim(-0.1,4.5)
        if(ifl==3): ax.set_ylim(-0.1,4.5)
        if(ifl==1): ax.set_ylim(-0.7,1.4)
        if(ifl==4): ax.set_ylim(-0.7,1.4)
        if(ifl==2): ax.set_ylim(-0.4,1.6)
        if(ifl==5): ax.set_ylim(-0.4,1.6)
                
    if(filelabel=="x0p00126"):
        if(ifl==0): ax.set_ylim(-0.1,7.0)
        if(ifl==3): ax.set_ylim(-0.1,7.0)
        if(ifl==1): ax.set_ylim(-0.2,1.8)
        if(ifl==4): ax.set_ylim(-0.2,1.8)
        if(ifl==2): ax.set_ylim(-0.2,1.7)
        if(ifl==5): ax.set_ylim(-0.7,1.7)
        
    ax.tick_params(which='both',direction='in',labelsize=12,right=True)
    ax.tick_params(which='major',length=7)
    ax.tick_params(which='minor',length=4)
    ax.set_ylabel(labelpdf[ifl],fontsize=16)
    if(ifl>5):
        ax.set_xlabel(r'$Q~({\rm GeV})$',fontsize=16)
    
    if(ifl==0):
        ax.text(0.10,0.85,stringx,fontsize=17,transform=ax.transAxes)

    # Add the legend
    if(ifl==0):
        ax.legend([(p1[0],p2[0]),(p3[0],p4[0])],\
                  #[ r"$A=208$", r"$A=208~(\rm wo~match)$" ], \
                  [ r"$A=208$",r"${\rm no~A~dep~in~NN}$" ], \
                  frameon=True,loc=4,prop={'size':13})

    icount = icount + 1

    # Add line indicating yadism region
    plt.axvline(5,lw=2,ls="dotted",color="black")
    if(ifl==2):
        ax.text(0.55,0.10,r'$\rm pQCD~constraints$',fontsize=14,transform=ax.transAxes,color="black")
        ax.arrow(x=0.55, y=0.05, dx=0.30, dy=0, width=.02,head_length=0.05,transform=ax.transAxes)
        

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('NNUSF-'+desc+'-'+filelabel+'.pdf')
print('output plot: NNUSF-'+desc+'-'+filelabel+'.pdf')

exit()
