
/*******************************************************************************
*
* File uflds.c
*
* Copyright (C) 2006, 2010-2016, 2021 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and initialization of the global gauge fields.
*
*   su3 *ufld(void)
*     Returns the base address of the single-precision gauge field. If it
*     is not already allocated, the field is allocated and initialized to
*     unity.
*
*   su3_dble *udfld(void)
*     Returns the base address of the double-precision gauge field. If it
*     is not already allocated, the field is allocated and initialized to
*     unity. Then the boundary conditions are set according to the data
*     base by calling set_bc() [bcnds.c].
*
*   void random_ud(void)
*     Initializes the active double-precision link variables to uniformly
*     distributed random SU(3) matrices. Then the boundary conditions are
*     set according to the data base by calling set_bc() [bcnds.c].
*
*   void set_ud_phase(void)
*     Multiplies the double-precision link variables U(x,k) by the phase
*     factor exp{i*theta[k-1]/N[k]}, for all k=1,2,3, where N[mu] is the
*     size of the (global) lattice in direction mu. The angles theta[0],
*     theta[1],theta[2] are set by set_bc_parms() [flags/lat_parms.c]. If
*     periodic boundary conditions are chosen in time, the variables U(x,0)
*     at global time N[0]-1 are multiplied by -1. The program does nothing
*     if the phase is already set according to the flags data base.
*
*   void unset_ud_phase(void)
*     Removes the phase of the double-precision link variables previously
*     set by set_ud_phase(). No action is performed if the phase is not
*     set according to the flags data base.
*
*   void renormalize_ud(void)
*     Projects the active double-precision link variables back to SU(3).
*     The static link variables are left untouched. An error occurs if
*     the phase of the field is set according to the flags data base [see
*     set_ud_phase() and unset_ud_phase()].
*
*   void assign_ud2u(void)
*     Assigns the double-precision gauge field to the single-precision
*     gauge field. All link variables in the local field, including the
*     static ones, are copied.
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

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


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
   int k,ix,mu,t;
   su3_mat_field *udv;

   if (udbv==NULL)
      alloc_udv();

   bc=bc_type();

#pragma omp parallel private(k,ix,mu,t,udv)
   {
      k=omp_get_thread_num();

∫      for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
      {
         int iy = ipt[ix];
         t=global_time(iy);
         for (mu=0;mu<4;mu++)
         {
            int iz = offset(iy, mu);
            if (mu==0)
            {
               if ((t!=(N0-1))||(bc!=0))
                  random_su3_mat_field(udbv, iz);
            }
            else
            {
               if ((t!=0)||(bc!=1))
                  random_su3_mat_field(udbv, iz);
            }
         }
      }
   }

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   #pragma omp target update \
    to(udbv->c1.base[0 : 6 * udbv->c1.volume]) \
    to(udbv->c2.base[0 : 6 * udbv->c2.volume]) \
    to(udbv->c3.base[0 : 6 * udbv->c3.volume])
}
