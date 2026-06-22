
/*******************************************************************************
*
* File geometry.c
*
* Copyright (C) 2005-2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the global lattice geometry.
*
*   void geometry(void)
*     Allocates and initializes all global index arrays that describe the
*     lattice geometry.
*
*   void ipt_global(int *x,int *ip,int *ix)
*     Given the Cartesian coordinates x[0],..,x[3] of a point on the full
*     lattice, this program finds the local lattice containing x. On exit
*     the rank of the associated MPI process is assigned to ip and the
*     local index of the point to ix.
*
*     Fills ipt for the thread block owned by thread k, whose origin in the
*     local lattice is (n0_ofs, n1_ofs, n2_ofs, n3_ofs).
*
*     The memory layout is time-major: for each time slice t, all spatial points
*     are grouped into cache blocks of size BLOCK_SIZE^3. Consecutive memory
*     indices within a time slice belong to the set
*
*     { (t, x1, x2, x3) : t fixed,
*                        x1 in [n1_ofs+BLOCK_SIZE*cb1, n1_ofs+BLOCK_SIZE*(cb1+1)),
*                        x2 in [n2_ofs+BLOCK_SIZE*cb2, n2_ofs+BLOCK_SIZE*(cb2+1)),
*                        x3 in [n3_ofs+BLOCK_SIZE*cb3, n3_ofs+BLOCK_SIZE*(cb3+1)) }
*
*     for each cache-block triple (cb1, cb2, cb3).
*
* Requires L1_TRD, L2_TRD and L3_TRD to be multiples of BLOCK_SIZE.
*
*   int global_time(int ix)
*     Returns the (global) time coordinate of the lattice point with local
*     index ix. If ix is out of range, NPROC0*L0 is returned.
*
* See main/README.global for a description of the lattice geometry.
*
* The program geometry() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously. All other programs are thread-safe and
* can be locally called, but assume that the index arrays have been set up by
* geometry().
*
*******************************************************************************/

#define GEOMETRY_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"

#pragma omp requires unified_shared_memory


int *tms=NULL;


static void alloc_ipt(void)
{
   ipt=malloc(VOLUME*sizeof(*ipt));

   error(ipt==NULL,1,"alloc_ipt [geometryv.c]",
         "Unable to allocate index array");
}


static void set_ipt(void)
{
   int y0,y1,y2,y3,lex;

   alloc_ipt();

   int mem=0;
   for (y0=0;y0<L0;y0++){
      for (y1=0;y1<L1;y1++){
         for (y2=0;y2<L2;y2++){
            for (y3=0;y3<L3;y3++){
               lex=y3+y2*L3+y1*L2*L3+y0*L1*L2*L3;
               ipt[lex]=mem;
               mem+=1;
            }
         }
      }
   }
   #pragma omp target enter data map(to : ipt[:VOLUME])
}


static void alloc_tms(void)
{
   tms=malloc(VOLUME*sizeof(*tms));

   error(tms==NULL,1,"alloc_tms [geometry.c]",
         "Unable to allocate time array");
   #pragma omp target enter data map(to : tms[:VOLUME])
}


static void set_tms(void)
{
   int k,ix,iy,x0;

   alloc_tms();

#pragma omp parallel private(k,ix,iy,x0)
   {
      k=omp_get_thread_num();

      for (iy=(k*VOLUME_TRD);iy<((k+1)*VOLUME_TRD);iy++)
      {
         x0=iy/(L1*L2*L3);
         ix=ipt[iy];

         tms[ix]=x0+cpr[0]*L0;
      }
   }
   #pragma omp target update to(tms[:VOLUME])
}


void geometry(void)
{
   printf("Using lexicogaphical layout. \n");
   if (ipt==NULL)
   {
      set_cpr();
      set_sbofs();
      set_ipt();
      set_iupdn();
      set_map();
      set_tms();
   }
}


void ipt_global(int *x,int *ip,int *ix)
{
   int x0,x1,x2,x3;
   int n[4];

   x0=safe_mod(x[0],NPROC0*L0);
   x1=safe_mod(x[1],NPROC1*L1);
   x2=safe_mod(x[2],NPROC2*L2);
   x3=safe_mod(x[3],NPROC3*L3);

   n[0]=x0/L0;
   n[1]=x1/L1;
   n[2]=x2/L2;
   n[3]=x3/L3;

   (*ip)=ipr_global(n);

   x0=x0%L0;
   x1=x1%L1;
   x2=x2%L2;
   x3=x3%L3;

   (*ix)=ipt[x3+x2*L3+x1*L2*L3+x0*L1*L2*L3];
}

#pragma omp declare target
int global_time(int ix)
{
   if ((ix>=0)&&(ix<VOLUME))
      return tms[ix];
      
   else
      return NPROC0*L0;
}
#pragma omp end declare target
