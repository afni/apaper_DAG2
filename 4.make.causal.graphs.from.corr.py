#!/usr/bin/env python

# A simple library (1 function!) for estimating the set of acyclic
# causal relations that can underly an observed set of correlations
# among 3 ROIs.
#
# This work comes from the following paper:
# 
#   Chen, G, Cai Z, Kording KP, Liu TT, Faskowitz J, Bandettini PA,
#   Biswal B, Taylor PA (2025). Resting-State fMRI and the Risk of
#   Overinterpretation: Noise, Mechanisms, and a Missing Rosetta Stone.
#   (submitted). https://doi.org/10.1101/2025.09.16.676611 
#
#
# author : PA Taylor (SSCC, NIMH, NIH, USA)
#          G Chen (SSCC, NIMH, NIH, USA
# ----------------------------------------------------------------------------
# date   : 17 Dec 2025
# ver    : 1.0
# ----------------------------------------------------------------------------

import sys
import numpy as np

# ============================================================================

def calc_acyclic_causal_rel_3x3(r13, r23, r12, verb=0):
    """For (nondirectionality) correlation values of 3 ROIs, estimate what
the possible (directional) acyclic causal relations ships among them
could be.  There are generally 6, with special cases of degeneracies
leading to fewer distinct solutions.

The input correlation values r?? can be pictured as describing the
correlation between 3 ROIs roi_? as follows:

                         roi_3
                         /  \\
                        /    \\
                  r13  /      \\ r23
                      /        \\
                     /          \\
                    /            \\
                 roi_1 -------- roi_2
                         r12

Parameters
----------
r13 : float
    correlation between roi_1 and roi_3
r23 : float
    correlation between roi_2 and roi_3
r23 : float
    correlation between roi_2 and roi_3
verb : int
    parameter to control verbosity of text output when running

Returns
-------
L_str_val : list of strings and values
    return a list of 3 strings describing directionality, and 3 
    corresponding floating point values, for the causal strengths

    """

    # check range of input values
    all_r = [r13,    r23,   r12]
    all_s = ['r13', 'r23', 'r12']
    for ii in range(len(all_r)):
        r = all_r[ii]
        s = all_s[ii]
        if np.abs(r) > 1.0 :
            msg = "** ERROR: corr value {} = {} ".format(s,r)
            msg+= "is outside allowed range [0.0, 1.0]"
            print(msg)
            sys.exit(-1)

    # sizes, related to dimensionality (AKA number of ROIs)
    N = 3
    P = 2*N

    # store the MRI corr values in this "nondirectional" NxN matrix;
    # translates 1-based counting of ROIs to 0-based indexing within Python
    R = np.zeros((N,N), dtype=float)
    R[0,1] = R[1,0] = r12
    R[1,2] = R[2,1] = r23
    R[0,2] = R[2,0] = r13

    if verb :
        print("++ R:")
        print(R)

    # ----- figure out matrices to store and manage permutations of indices

    # initialize matrices with permutations of ROIs
    M1 = np.zeros((N,N), dtype=int)
    M2 = np.zeros((N,N), dtype=int)
    M  = np.zeros((P,N), dtype=int)
    
    # initialize list to hold strings for reporting ROI directionality
    S  = [[[], [], []], [[], [], []], [[], [], []]]

    # initialize matrix to hold estimated values
    C  = np.zeros((P,N,N), dtype=float)

    # create sub-matrices of permuted indices; also prepare str for reporting
    for ii in range(N):
        for jj in range(N):
            M1[ii,jj]+= (jj + ii*2) % N
            M2[jj,ii]+= (jj + ii*2) % N
            S[ii][jj] = "roi_{} -> roi_{}".format(ii+1,jj+1)

    # create final array of permutation indices
    for ii in range(N):
        for jj in range(N):
            aa = (M1[ii,0]+M1[ii,2]-1)*(N-1)
            bb = (M2[ii,0]+M2[ii,2]-1)*(N-1)+1
            M[aa,jj] = M1[ii,jj]
            M[bb,jj] = M2[ii,jj]

    if verb :
        print("++ M:")
        print(M)

    # ----- do actual work

    # go through each permutation and do calcs
    for pp in range(P):
        
        # get our indices for this permutation
        ii,jj,kk = M[pp,:]

        if verb :
            print("++ indices :", ii,jj,kk)

        # this is the "giver" -> "bridge" ROI connection
        C[pp,ii,kk] = R[ii,kk]

        # ... which also determines a scaling factor
        denom = 1.0 - R[ii,kk]**2
        if verb :
            print("++ denom :", denom)

        # now calculate the other two edges

        if denom :
            C[pp,ii,jj] = (R[ii,jj] - R[ii,kk]*R[kk,jj])/denom
            C[pp,kk,jj] = (R[kk,jj] - R[ii,kk]*R[ii,jj])/denom
        else:
            C[pp,ii,jj] = 0.0
            C[pp,kk,jj] = 0.0

        # print out; the directionality of what indices are matters
        print("")
        print("   {}  : {:5.2f}".format(S[ii][kk], C[pp,ii,kk]))
        print("   {}  : {:5.2f}".format(S[ii][jj], C[pp,ii,jj]))
        print("   {}  : {:5.2f}".format(S[kk][jj], C[pp,kk,jj]))

    # return the list of string (directional info) and values
    L_str_val = [S[ii][kk],   S[ii][jj],   S[kk][jj], \
                 C[pp,ii,kk], C[pp,ii,jj], C[pp,kk,jj] ]
    return L_str_val

# ===============================================================


if __name__ == "__main__" :
    
    # first example
    print("=" * 60)
    print("++ Example from Fig. 9B")
    r13 =  0.6
    r23 = -0.5
    r12 = -0.05
    L_str_val_ex9B = calc_acyclic_causal_rel_3x3(r13, r23, r12)

    # second example
    print("=" * 60)
    print("++ Example from Fig. 9C")
    r13 = 0.8
    r23 = 0.75
    r12 = 0.6
    L_str_val_ex9C = calc_acyclic_causal_rel_3x3(r13, r23, r12)

