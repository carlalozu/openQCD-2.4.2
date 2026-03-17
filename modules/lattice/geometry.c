
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

int *tms=NULL;


static void alloc_ipt(void)
{
   ipt=malloc(VOLUME*sizeof(*ipt));

   error(ipt==NULL,1,"alloc_ipt [geometry.c]",
         "Unable to allocate index array");
}


static void set_ipt(void)
{
   int k,n0,n1,n2,n3,ix;
   int ofs;

   alloc_ipt();

#pragma omp parallel private(k,n0,n1,n2,n3,ix)
   {
      k=omp_get_thread_num();

      ofs = k*VOLUME_TRD;
      // ix=n3+n2*L3_TRD+L1_TRD*L2_TRD*L3_TRD+n0*L1_TRD*L2_TRD*L3_TRD;
      for (n0=0;n0<L0_TRD;n0++)
      {
         for (n1=0;n1<L1_TRD;n1++)
         {
            for (n2=0;n2<L2_TRD;n2++)
            {
               for (n3=0;n3<L3_TRD;n3++)
               {
                  ipt[ofs]=ofs;
                  ofs+=1;
               }
            }
         }
      }
   }
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
