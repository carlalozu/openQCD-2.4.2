
/*******************************************************************************
*
* File plaq_sum.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Calculation of plaquette sums.
*
*   double plaq_sum_dble(int icom)
*     Returns the sum of Re[tr{U(p)}] over all unoriented plaquettes p,
*     where U(p) is the product of the double-precision link variables
*     around p. If icom=1 the global sum of the local sums is returned
*     and otherwise just the local sum.
*
*   double plaq_wsum_dble(int icom)
*     Same as plaq_sum_dble(), but giving weight 1/2 to the contribution
*     of the space-like plaquettes at the boundaries of the lattice if
*     boundary conditions of type 0,1 or 2 are chosen. If icom=1 the global
*     sum of the local sums is returned and otherwise just the local sum.
*
*
* The Wilson plaquette action density is defined so that it converges to the
* Yang-Mills action in the classical continuum limit with a rate proportional
* to a^2. In particular, at the boundaries of the lattice (if there are any),
* the space-like plaquettes are given the weight 1/2 and the contribution of
* a plaquette p in the bulk is 2*Re[tr{1-U(p)}].
*
* The time-slice sum asl[x0] computed by plaq_action_slices() includes the
* full contribution to the action of the space-like plaquettes at time x0 and
* 1/2 of the contribution of the time-like plaquettes at time x0 and x0-1.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously. It is taken for granted that
* the lattice geometry index arrays have been set up.
*
*******************************************************************************/

#define PLAQ_SUM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "su3fcts.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)

static double *qsm[2*N0];
static qflt rqsmE[N0],rqsmB[N0];
static su3_dble *udb;
static su3_mat_field *udbv;
prof_section compute = {.name = "local_plaq_sum_dble"};




#pragma omp declare target
static double plaq_dblev(su3_mat_field *udbv,int n,int ix)
{
   double sm;
   int ip[4];
   su3_dble wd1 ALIGNED16;
   su3_dble wd2 ALIGNED16;

   plaq_uidx(n,ix,ip);

   fsu3matxsu3mat(udbv, &wd1, ip[0], ip[1]);
   fsu3matdagxsu3matdag(udbv, &wd2, ip[3], ip[2]);
   cm3x3_retr(&wd1,&wd2,&sm);

   return sm;

double local_plaq_dblev(int n){
   udbv=udfldv();
   double pa;
   #pragma omp parallel reduction(+:pa)
   {
      int k=omp_get_thread_num();
      for (int ix=(k*VOLUME_TRD);ix<((k+1)*VOLUME_TRD);ix++)
         pa += plaq_dblev(n, ix);
   }
   return pa;
}


static qflt local_plaq_sum_dble(int iw)
{
   double wp,pa=0.0;
   qflt rqsm;

   int bc=bc_type();

   if (iw==0)
      wp=1.0;
   else
      wp=0.5;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;
   udbv=udfldv();
   prof_begin(&compute);
   // #pragma omp parallel private(k,ix,t,n,pa) reduction(sum_qflt : rqsm)
   #pragma omp target teams distribute parallel for reduction(+:pa)
   for (int ix=0;ix<VOLUME;ix++)
   {
      double local_pa=0.0;
      int t=global_time(ix);
      if ((t<(N0-1))||(bc!=0))
      {
         for (int n=0;n<3;n++)
            local_pa+=plaq_dblev(udbv,n,ix);
      }
      
      if (((t>0)&&(t<(N0-1)))||(bc==3))
      {
         for (int n=3;n<6;n++)
            local_pa+=plaq_dblev(udbv,n,ix);
      }
      else if ((t==0)||(bc==0))
      {
         if (bc==1)
            local_pa+=wp*9.0;
         else
         {
            for (int n=3;n<6;n++)
               local_pa+=wp*plaq_dblev(udbv,n,ix);
         }
      }
      else
      {
         for (int n=3;n<6;n++)
            local_pa+=plaq_dblev(udbv,n,ix);

         local_pa+=wp*9.0;
      }
      pa += local_pa;
   }
   prof_end(&compute);
   acc_qflt(pa,rqsm.q);
   return rqsm;
}


double plaq_sum_dblev(int icom)
{
   qflt rqsm;

   set_uidx();

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dblev(0);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


double plaq_wsum_dblev(int icom)
{
   qflt rqsm;

   set_uidx();

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dblev(1);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}

