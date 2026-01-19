
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
*   double plaq_action_slices(double *asl)
*     Computes the time-slice sums asl[x0] of the tree-level O(a)-improved
*     plaquette action density of the double-precision gauge field. The
*     factor 1/g0^2 is omitted and the time x0 runs from 0 to NPROC0*L0-1.
*     The program returns the total action.
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
#pragma omp declare target
static su3_dble *udb;
static int plaq_uidx_array[6*VOLUME*4];
static qflt rqsm;
#pragma omp end declare target

static double plaq_dble(int n,int ix)
{
      int *ip = &plaq_uidx_array[n*VOLUME*4 + ix*4];
      double sm;
      su3_dble wd1;
      su3_dble wd2;

      //su3xsu3(&udb_test[ip[0]],&udb_test[ip[1]],&wd1);
      //su3dagxsu3dag(&udb_test[ip[3]],&udb_test[ip[2]], &wd2);
      su3xsu3(udb+ip[0],udb+ip[1],&wd1);
      su3dagxsu3dag(udb+ip[3],udb+ip[2],&wd2);
      cm3x3_retr(&wd1,&wd2,&sm);

      return sm; 
}

static double plaq_dble_custom(int udb_index)
{
   double sm = 0.0; 

   sm+=(*(udb+udb_index)).c11.re + (*(udb+udb_index)).c11.im;
   sm+=(*(udb+udb_index)).c12.re + (*(udb+udb_index)).c12.im;
   sm+=(*(udb+udb_index)).c13.re + (*(udb+udb_index)).c13.im;
   sm+=(*(udb+udb_index)).c21.re + (*(udb+udb_index)).c21.im;
   sm+=(*(udb+udb_index)).c22.re + (*(udb+udb_index)).c22.im;
   sm+=(*(udb+udb_index)).c23.re + (*(udb+udb_index)).c23.im;
   sm+=(*(udb+udb_index)).c31.re + (*(udb+udb_index)).c31.im;
   sm+=(*(udb+udb_index)).c32.re + (*(udb+udb_index)).c32.im;
   sm+=(*(udb+udb_index)).c33.re + (*(udb+udb_index)).c33.im;

   return sm; 
}

static double plaq_dble_cm3x3_retr_indices(int udb_index_u, int udb_index_v)
{
   double r = 0.0;

   r =(*(udb+udb_index_u)).c11.re*(*(udb+udb_index_v)).c11.re-(*(udb+udb_index_u)).c11.im*(*(udb+udb_index_v)).c11.im;
   r+=(*(udb+udb_index_u)).c12.re*(*(udb+udb_index_v)).c21.re-(*(udb+udb_index_u)).c12.im*(*(udb+udb_index_v)).c21.im;
   r+=(*(udb+udb_index_u)).c13.re*(*(udb+udb_index_v)).c31.re-(*(udb+udb_index_u)).c13.im*(*(udb+udb_index_v)).c31.im;
   r+=(*(udb+udb_index_u)).c21.re*(*(udb+udb_index_v)).c12.re-(*(udb+udb_index_u)).c21.im*(*(udb+udb_index_v)).c12.im;
   r+=(*(udb+udb_index_u)).c22.re*(*(udb+udb_index_v)).c22.re-(*(udb+udb_index_u)).c22.im*(*(udb+udb_index_v)).c22.im;
   r+=(*(udb+udb_index_u)).c23.re*(*(udb+udb_index_v)).c32.re-(*(udb+udb_index_u)).c23.im*(*(udb+udb_index_v)).c32.im;
   r+=(*(udb+udb_index_u)).c31.re*(*(udb+udb_index_v)).c13.re-(*(udb+udb_index_u)).c31.im*(*(udb+udb_index_v)).c13.im;
   r+=(*(udb+udb_index_u)).c32.re*(*(udb+udb_index_v)).c23.re-(*(udb+udb_index_u)).c32.im*(*(udb+udb_index_v)).c23.im;
   r+=(*(udb+udb_index_u)).c33.re*(*(udb+udb_index_v)).c33.re-(*(udb+udb_index_u)).c33.im*(*(udb+udb_index_v)).c33.im;

   return r;
}

static double plaq_dble_cm3x3_retr(su3_dble *u, su3_dble *v)
{
   double r = 0.0;

   r =(*(u)).c11.re*(*(v)).c11.re-(*(u)).c11.im*(*(v)).c11.im;
   r+=(*(u)).c12.re*(*(v)).c21.re-(*(u)).c12.im*(*(v)).c21.im;
   r+=(*(u)).c13.re*(*(v)).c31.re-(*(u)).c13.im*(*(v)).c31.im;
   r+=(*(u)).c21.re*(*(v)).c12.re-(*(u)).c21.im*(*(v)).c12.im;
   r+=(*(u)).c22.re*(*(v)).c22.re-(*(u)).c22.im*(*(v)).c22.im;
   r+=(*(u)).c23.re*(*(v)).c32.re-(*(u)).c23.im*(*(v)).c32.im;
   r+=(*(u)).c31.re*(*(v)).c13.re-(*(u)).c31.im*(*(v)).c13.im;
   r+=(*(u)).c32.re*(*(v)).c23.re-(*(u)).c32.im*(*(v)).c23.im;
   r+=(*(u)).c33.re*(*(v)).c33.re-(*(u)).c33.im*(*(v)).c33.im;

   return r;
}

static su3_dble plaq_dble_su3xsu3(su3_dble *u,su3_dble *v)
{
   su3_vector_dble psi,chi;
   su3_dble w;

   psi.c1=(*v).c11;
   psi.c2=(*v).c21;
   psi.c3=(*v).c31;

   (chi).c1.re= (*(u)).c11.re*(psi).c1.re-(*(u)).c11.im*(psi).c1.im + (*(u)).c12.re*(psi).c2.re-(*(u)).c12.im*(psi).c2.im + (*(u)).c13.re*(psi).c3.re-(*(u)).c13.im*(psi).c3.im;
   (chi).c1.im= (*(u)).c11.re*(psi).c1.im+(*(u)).c11.im*(psi).c1.re + (*(u)).c12.re*(psi).c2.im+(*(u)).c12.im*(psi).c2.re + (*(u)).c13.re*(psi).c3.im+(*(u)).c13.im*(psi).c3.re;
   (chi).c2.re= (*(u)).c21.re*(psi).c1.re-(*(u)).c21.im*(psi).c1.im + (*(u)).c22.re*(psi).c2.re-(*(u)).c22.im*(psi).c2.im + (*(u)).c23.re*(psi).c3.re-(*(u)).c23.im*(psi).c3.im;
   (chi).c2.im= (*(u)).c21.re*(psi).c1.im+(*(u)).c21.im*(psi).c1.re + (*(u)).c22.re*(psi).c2.im+(*(u)).c22.im*(psi).c2.re + (*(u)).c23.re*(psi).c3.im+(*(u)).c23.im*(psi).c3.re;
   (chi).c3.re= (*(u)).c31.re*(psi).c1.re-(*(u)).c31.im*(psi).c1.im + (*(u)).c32.re*(psi).c2.re-(*(u)).c32.im*(psi).c2.im + (*(u)).c33.re*(psi).c3.re-(*(u)).c33.im*(psi).c3.im;
   (chi).c3.im= (*(u)).c31.re*(psi).c1.im+(*(u)).c31.im*(psi).c1.re + (*(u)).c32.re*(psi).c2.im+(*(u)).c32.im*(psi).c2.re + (*(u)).c33.re*(psi).c3.im+(*(u)).c33.im*(psi).c3.re;

   w.c11=chi.c1;
   w.c21=chi.c2;
   w.c31=chi.c3;

   psi.c1=(*v).c12;
   psi.c2=(*v).c22;
   psi.c3=(*v).c32;

   (chi).c1.re= (*(u)).c11.re*(psi).c1.re-(*(u)).c11.im*(psi).c1.im + (*(u)).c12.re*(psi).c2.re-(*(u)).c12.im*(psi).c2.im + (*(u)).c13.re*(psi).c3.re-(*(u)).c13.im*(psi).c3.im;
   (chi).c1.im= (*(u)).c11.re*(psi).c1.im+(*(u)).c11.im*(psi).c1.re + (*(u)).c12.re*(psi).c2.im+(*(u)).c12.im*(psi).c2.re + (*(u)).c13.re*(psi).c3.im+(*(u)).c13.im*(psi).c3.re;
   (chi).c2.re= (*(u)).c21.re*(psi).c1.re-(*(u)).c21.im*(psi).c1.im + (*(u)).c22.re*(psi).c2.re-(*(u)).c22.im*(psi).c2.im + (*(u)).c23.re*(psi).c3.re-(*(u)).c23.im*(psi).c3.im;
   (chi).c2.im= (*(u)).c21.re*(psi).c1.im+(*(u)).c21.im*(psi).c1.re + (*(u)).c22.re*(psi).c2.im+(*(u)).c22.im*(psi).c2.re + (*(u)).c23.re*(psi).c3.im+(*(u)).c23.im*(psi).c3.re;
   (chi).c3.re= (*(u)).c31.re*(psi).c1.re-(*(u)).c31.im*(psi).c1.im + (*(u)).c32.re*(psi).c2.re-(*(u)).c32.im*(psi).c2.im + (*(u)).c33.re*(psi).c3.re-(*(u)).c33.im*(psi).c3.im;
   (chi).c3.im= (*(u)).c31.re*(psi).c1.im+(*(u)).c31.im*(psi).c1.re + (*(u)).c32.re*(psi).c2.im+(*(u)).c32.im*(psi).c2.re + (*(u)).c33.re*(psi).c3.im+(*(u)).c33.im*(psi).c3.re;

   w.c12=chi.c1;
   w.c22=chi.c2;
   w.c32=chi.c3;

   psi.c1=(*v).c13;
   psi.c2=(*v).c23;
   psi.c3=(*v).c33;

   (chi).c1.re= (*(u)).c11.re*(psi).c1.re-(*(u)).c11.im*(psi).c1.im + (*(u)).c12.re*(psi).c2.re-(*(u)).c12.im*(psi).c2.im + (*(u)).c13.re*(psi).c3.re-(*(u)).c13.im*(psi).c3.im;
   (chi).c1.im= (*(u)).c11.re*(psi).c1.im+(*(u)).c11.im*(psi).c1.re + (*(u)).c12.re*(psi).c2.im+(*(u)).c12.im*(psi).c2.re + (*(u)).c13.re*(psi).c3.im+(*(u)).c13.im*(psi).c3.re;
   (chi).c2.re= (*(u)).c21.re*(psi).c1.re-(*(u)).c21.im*(psi).c1.im + (*(u)).c22.re*(psi).c2.re-(*(u)).c22.im*(psi).c2.im + (*(u)).c23.re*(psi).c3.re-(*(u)).c23.im*(psi).c3.im;
   (chi).c2.im= (*(u)).c21.re*(psi).c1.im+(*(u)).c21.im*(psi).c1.re + (*(u)).c22.re*(psi).c2.im+(*(u)).c22.im*(psi).c2.re + (*(u)).c23.re*(psi).c3.im+(*(u)).c23.im*(psi).c3.re;
   (chi).c3.re= (*(u)).c31.re*(psi).c1.re-(*(u)).c31.im*(psi).c1.im + (*(u)).c32.re*(psi).c2.re-(*(u)).c32.im*(psi).c2.im + (*(u)).c33.re*(psi).c3.re-(*(u)).c33.im*(psi).c3.im;
   (chi).c3.im= (*(u)).c31.re*(psi).c1.im+(*(u)).c31.im*(psi).c1.re + (*(u)).c32.re*(psi).c2.im+(*(u)).c32.im*(psi).c2.re + (*(u)).c33.re*(psi).c3.im+(*(u)).c33.im*(psi).c3.re;

   w.c13=chi.c1;
   w.c23=chi.c2;
   w.c33=chi.c3;

   return w; 
}

static su3_dble plaq_dble_su3dagxsu3dag(su3_dble *u,su3_dble *v)
{
   su3_vector_dble psi,chi;
   su3_dble w;

   psi.c1.re= (*v).c11.re;
   psi.c1.im=-(*v).c11.im;
   psi.c2.re= (*v).c12.re;
   psi.c2.im=-(*v).c12.im;
   psi.c3.re= (*v).c13.re;
   psi.c3.im=-(*v).c13.im;

   (chi).c1.re= (*u).c11.re*(psi).c1.re+(*u).c11.im*(psi).c1.im + (*u).c21.re*(psi).c2.re+(*u).c21.im*(psi).c2.im + (*u).c31.re*(psi).c3.re+(*u).c31.im*(psi).c3.im;
   (chi).c1.im= (*u).c11.re*(psi).c1.im-(*u).c11.im*(psi).c1.re + (*u).c21.re*(psi).c2.im-(*u).c21.im*(psi).c2.re + (*u).c31.re*(psi).c3.im-(*u).c31.im*(psi).c3.re;
   (chi).c2.re= (*u).c12.re*(psi).c1.re+(*u).c12.im*(psi).c1.im + (*u).c22.re*(psi).c2.re+(*u).c22.im*(psi).c2.im + (*u).c32.re*(psi).c3.re+(*u).c32.im*(psi).c3.im;
   (chi).c2.im= (*u).c12.re*(psi).c1.im-(*u).c12.im*(psi).c1.re + (*u).c22.re*(psi).c2.im-(*u).c22.im*(psi).c2.re + (*u).c32.re*(psi).c3.im-(*u).c32.im*(psi).c3.re;
   (chi).c3.re= (*u).c13.re*(psi).c1.re+(*u).c13.im*(psi).c1.im + (*u).c23.re*(psi).c2.re+(*u).c23.im*(psi).c2.im + (*u).c33.re*(psi).c3.re+(*u).c33.im*(psi).c3.im;
   (chi).c3.im= (*u).c13.re*(psi).c1.im-(*u).c13.im*(psi).c1.re + (*u).c23.re*(psi).c2.im-(*u).c23.im*(psi).c2.re + (*u).c33.re*(psi).c3.im-(*u).c33.im*(psi).c3.re;

   w.c11=chi.c1;
   w.c21=chi.c2;
   w.c31=chi.c3;

   psi.c1.re= (*v).c21.re;
   psi.c1.im=-(*v).c21.im;
   psi.c2.re= (*v).c22.re;
   psi.c2.im=-(*v).c22.im;
   psi.c3.re= (*v).c23.re;
   psi.c3.im=-(*v).c23.im;
   
   (chi).c1.re= (*u).c11.re*(psi).c1.re+(*u).c11.im*(psi).c1.im + (*u).c21.re*(psi).c2.re+(*u).c21.im*(psi).c2.im + (*u).c31.re*(psi).c3.re+(*u).c31.im*(psi).c3.im;
   (chi).c1.im= (*u).c11.re*(psi).c1.im-(*u).c11.im*(psi).c1.re + (*u).c21.re*(psi).c2.im-(*u).c21.im*(psi).c2.re + (*u).c31.re*(psi).c3.im-(*u).c31.im*(psi).c3.re;
   (chi).c2.re= (*u).c12.re*(psi).c1.re+(*u).c12.im*(psi).c1.im + (*u).c22.re*(psi).c2.re+(*u).c22.im*(psi).c2.im + (*u).c32.re*(psi).c3.re+(*u).c32.im*(psi).c3.im;
   (chi).c2.im= (*u).c12.re*(psi).c1.im-(*u).c12.im*(psi).c1.re + (*u).c22.re*(psi).c2.im-(*u).c22.im*(psi).c2.re + (*u).c32.re*(psi).c3.im-(*u).c32.im*(psi).c3.re;
   (chi).c3.re= (*u).c13.re*(psi).c1.re+(*u).c13.im*(psi).c1.im + (*u).c23.re*(psi).c2.re+(*u).c23.im*(psi).c2.im + (*u).c33.re*(psi).c3.re+(*u).c33.im*(psi).c3.im;
   (chi).c3.im= (*u).c13.re*(psi).c1.im-(*u).c13.im*(psi).c1.re + (*u).c23.re*(psi).c2.im-(*u).c23.im*(psi).c2.re + (*u).c33.re*(psi).c3.im-(*u).c33.im*(psi).c3.re;

   w.c12=chi.c1;
   w.c22=chi.c2;
   w.c32=chi.c3;

   psi.c1.re= (*v).c31.re;
   psi.c1.im=-(*v).c31.im;
   psi.c2.re= (*v).c32.re;
   psi.c2.im=-(*v).c32.im;
   psi.c3.re= (*v).c33.re;
   psi.c3.im=-(*v).c33.im;
   
   (chi).c1.re= (*u).c11.re*(psi).c1.re+(*u).c11.im*(psi).c1.im + (*u).c21.re*(psi).c2.re+(*u).c21.im*(psi).c2.im + (*u).c31.re*(psi).c3.re+(*u).c31.im*(psi).c3.im;
   (chi).c1.im= (*u).c11.re*(psi).c1.im-(*u).c11.im*(psi).c1.re + (*u).c21.re*(psi).c2.im-(*u).c21.im*(psi).c2.re + (*u).c31.re*(psi).c3.im-(*u).c31.im*(psi).c3.re;
   (chi).c2.re= (*u).c12.re*(psi).c1.re+(*u).c12.im*(psi).c1.im + (*u).c22.re*(psi).c2.re+(*u).c22.im*(psi).c2.im + (*u).c32.re*(psi).c3.re+(*u).c32.im*(psi).c3.im;
   (chi).c2.im= (*u).c12.re*(psi).c1.im-(*u).c12.im*(psi).c1.re + (*u).c22.re*(psi).c2.im-(*u).c22.im*(psi).c2.re + (*u).c32.re*(psi).c3.im-(*u).c32.im*(psi).c3.re;
   (chi).c3.re= (*u).c13.re*(psi).c1.re+(*u).c13.im*(psi).c1.im + (*u).c23.re*(psi).c2.re+(*u).c23.im*(psi).c2.im + (*u).c33.re*(psi).c3.re+(*u).c33.im*(psi).c3.im;
   (chi).c3.im= (*u).c13.re*(psi).c1.im-(*u).c13.im*(psi).c1.re + (*u).c23.re*(psi).c2.im-(*u).c23.im*(psi).c2.re + (*u).c33.re*(psi).c3.im-(*u).c33.im*(psi).c3.re;

   w.c13=chi.c1;
   w.c23=chi.c2;
   w.c33=chi.c3;

   return w; 
}

static double plaq_dble_original(int n,int ix)
{
   int *ip = &plaq_uidx_array[n*VOLUME*4 + ix*4];
   double sm;
   su3_dble wd1;// = __runtime_alloc(18*sizeof(double));// ALIGNED16;
   su3_dble wd2;// = __runtime_alloc(18*sizeof(double));// ALIGNED16;

   wd1 = su3xsu3_test(udb+ip[0],udb+ip[1]);
   wd2 = su3dagxsu3dag_test(udb+ip[3],udb+ip[2]);
   cm3x3_retr(&wd1,&wd2,&sm);

   return sm;
}

static qflt local_plaq_sum_dble(int iw)
{
   int bc,k,ix,t,n;
   double wp,pa,time_start,time_end;
   qflt rqsm;
   int mysize = 4*VOLUME+7*(BNDRY/4); 
   
   bc=bc_type();

   if (iw==0)
      wp=1.0;
   else
      wp=0.5;


   udb=udfld();
   set_plaq_uidx_array((plaq_uidx_array));

   //int *ip_test = &plaq_uidx_array[3*VOLUME*4 + 20000*4];
   //printf("plaq_uidx_array[0] = %d\n",ip_test[0]);
   //printf("plaq_uidx_array[1] = %d\n",ip_test[1]);
   //printf("plaq_uidx_array[2] = %d\n",ip_test[2]);
   //printf("plaq_uidx_array[3] = %d\n\n",ip_test[3]);


   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;

   pa = 0.0;

   #pragma omp target update to(plaq_uidx_array)

   #pragma omp target data map(to:udb[0:4*VOLUME+7*(BNDRY/4)])
   {
      MPI_Barrier(MPI_COMM_WORLD);
      time_start=MPI_Wtime();

      //#pragma omp parallel for reduction(+ : pa) 
      #pragma omp target teams distribute parallel for reduction(+ : pa) thread_limit(8)
      {
         for (int udb_index = 0; udb_index < VOLUME; udb_index++)
         {
            for (size_t n = 0; n < 6; n++)
            {
               int *ip_test = &plaq_uidx_array[n*VOLUME*4 + udb_index*4];
               su3_dble wd1 = plaq_dble_su3xsu3(udb+ip_test[0],udb+ip_test[1]);
               su3_dble wd2 = plaq_dble_su3dagxsu3dag(udb+ip_test[3],udb+ip_test[2]);
               pa += plaq_dble_cm3x3_retr(&wd1, &wd2);
            }
         }    
      }
   }

   acc_qflt(pa,rqsm.q);

   MPI_Barrier(MPI_COMM_WORLD);
   time_end=MPI_Wtime();
   plaq_time += time_end-time_start; 

   return rqsm;
}


double plaq_sum_dble(int icom)
{
   qflt rqsm;

   set_uidx();

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dble(0);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


double plaq_wsum_dble(int icom)
{
   qflt rqsm;

   set_uidx();

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dble(1);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


double plaq_action_slices(double *asl)
{
   int bc,k,ix,t,n;
   double smE,smB;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();
   else
      set_uidx();

   bc=bc_type();
   udb=udfld();

   for (t=0;t<N0;t++)
   {
      qsm[t]=rqsmE[t].q;
      rqsmE[t].q[0]=0.0;
      rqsmE[t].q[1]=0.0;

      qsm[t+N0]=rqsmB[t].q;
      rqsmB[t].q[0]=0.0;
      rqsmB[t].q[1]=0.0;
   }

#pragma omp parallel private(k,ix,t,n,smE,smB) \
   reduction(sum_qflt : rqsmE[cpr[0]*L0:L0],rqsmB[cpr[0]*L0:L0])
   {
      k=omp_get_thread_num();

      for (ix=(k*VOLUME_TRD);ix<((k+1)*VOLUME_TRD);ix++)
      {
         t=global_time(ix);
         smE=0.0;
         smB=0.0;

         if ((t<(N0-1))||(bc!=0))
         {
            for (n=0;n<3;n++)
               smE+=(3.0-plaq_dble(n,ix));
         }

         if ((t>0)||(bc!=1))
         {
            for (n=3;n<6;n++)
               smB+=(3.0-plaq_dble(n,ix));
         }

         acc_qflt(smE,rqsmE[t].q);
         acc_qflt(smB,rqsmB[t].q);
      }
   }

   global_qsum(2*N0,qsm,qsm);

   if (bc!=3)
      add_qflt(rqsmE[0].q,rqsmB[0].q,rqsmB[0].q);
   else
   {
      rqsmB[0].q[0]*=2.0;
      rqsmB[0].q[1]*=2.0;
      add_qflt(rqsmE[0].q,rqsmB[0].q,rqsmB[0].q);
      add_qflt(rqsmE[N0-1].q,rqsmB[0].q,rqsmB[0].q);
   }

   if (bc==0)
   {
      for (t=1;t<(N0-1);t++)
      {
         rqsmB[t].q[0]*=2.0;
         rqsmB[t].q[1]*=2.0;
         add_qflt(rqsmE[t].q,rqsmB[t].q,rqsmB[t].q);
         add_qflt(rqsmE[t-1].q,rqsmB[t].q,rqsmB[t].q);
      }

      add_qflt(rqsmE[N0-2].q,rqsmB[N0-1].q,rqsmB[N0-1].q);
   }
   else
   {
      for (t=1;t<N0;t++)
      {
         rqsmB[t].q[0]*=2.0;
         rqsmB[t].q[1]*=2.0;
         add_qflt(rqsmE[t].q,rqsmB[t].q,rqsmB[t].q);
         add_qflt(rqsmE[t-1].q,rqsmB[t].q,rqsmB[t].q);
      }
   }

   for (t=0;t<N0;t++)
   {
      asl[t]=rqsmB[t].q[0];

      if (t>0)
         add_qflt(rqsmB[t].q,rqsmB[0].q,rqsmB[0].q);
   }

   if ((bc==1)||(bc==2))
      add_qflt(rqsmE[N0-1].q,rqsmB[0].q,rqsmB[0].q);

   return rqsmB[0].q[0];
}
