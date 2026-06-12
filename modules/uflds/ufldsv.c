
/*******************************************************************************
*
* File ufldsv.c
*
* Copyright (C) 2006, 2010-2016, 2021 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and initialization of the global gauge field in SoA vector layout.
*
*   su3_mat_field *udfldv(void)
*     Returns a pointer to the double-precision gauge field in SoA vector
*     layout. If it is not already allocated, the field is allocated and
*     initialized to unity. Then the boundary conditions are set according
*     to the data base by calling set_bc() [bcnds.c].
*
*   void random_udv(void)
*     Initializes the active double-precision link variables to uniformly
*     distributed random SU(3) matrices. Then the boundary conditions are
*     set according to the data base by calling set_bc() [bcnds.c].
*
*   void update_su3_mat_field(void)
*     Update values of su3_mat_field on the GPU.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously. It is taken for granted that
* the lattice geometry index arrays have been set up.
*
*******************************************************************************/

#define UFLDSV_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

static su3_mat_field *udbv=NULL;

static void alloc_udv(void)
{
   int bc;
   size_t n;

   error_root(sizeof(su3_dble)!=(18*sizeof(double)),1,"alloc_ud [uflds.c]",
              "The su3_dble structures are not properly packed");

   error(ipt==NULL,1,"alloc_ud [uflds.c]","Geometry arrays are not set");

   bc=bc_type();
   n=4*VOLUME+7*(BNDRY/4);

   if ((cpr[0]==(NPROC0-1))&&((bc==1)||(bc==2)))
      n+=3;

   udbv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
   su3_mat_field_init(udbv, n);
   error(udbv==NULL,1,"alloc_udv [uflds.c]",
         "Unable to allocate memory space for the gauge field");
   set_ud2unityv(4*VOLUME_TRD,2,udbv);

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   enter_su3_mat_field(udbv);
}


su3_mat_field *udfldv(void)
{
   if (udbv==NULL)
      alloc_udv();

   return udbv;
}

void random_udv(void)
{
   int bc;
   int k,ix,iz,t;
   su3_mat_field *udv;

   if (udbv==NULL)
      alloc_udv();

   bc=bc_type();

#pragma omp parallel private(k,ix,iz,t,udv)
   {
      k=omp_get_thread_num();
      for (int iy=k*VOLUME_TRD;iy<(k+1)*VOLUME_TRD;iy++){
         for (int mu=0;mu<4;mu++) {
            ix=ipt[iy];
            t=global_time(ix);
            iz=offset(ix,mu);
            random_su3_mat_field(udbv, iz);
         }
      }
   }

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
}

void update_su3_mat_field(void)
{
   if (udbv!=NULL)
   {
      #pragma omp target update \
      to(udbv->c1.base[0 : 6 * udbv->c1.volume]) \
      to(udbv->c2.base[0 : 6 * udbv->c2.volume]) \
      to(udbv->c3.base[0 : 6 * udbv->c3.volume])
   }
   else
      printf("udbv hasn't been initialized");
}
