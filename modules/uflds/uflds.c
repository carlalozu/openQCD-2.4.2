
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

#define UFLDS_C

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

typedef union
{
   su3 u;
   float r[18];
} umat_t;

typedef union
{
   su3_dble u;
   double r[18];
} umat_dble_t;

static complex_dble phase[3] ALIGNED16 ={{0.0,0.0}};
static su3 *ub=NULL;
static su3_dble *udb=NULL;
static su3_mat_field *udbv=NULL;


static void alloc_u(void)
{
   size_t n;

   error_root(sizeof(su3)!=(18*sizeof(float)),1,"alloc_u [uflds.c]",
              "The su3 structures are not properly packed");

   n=4*VOLUME;
   ub=amalloc(n*sizeof(*ub),ALIGN);
   error(ub==NULL,1,"alloc_u [uflds.c]",
         "Unable to allocate memory space for the gauge field");
   set_u2unity(4*VOLUME_TRD,2,ub);

   set_flags(UPDATED_U);
}


su3 *ufld(void)
{
   if (ub==NULL)
      alloc_u();

   return ub;
}


static void alloc_ud(void)
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

   udb=amalloc(n*sizeof(*udb),ALIGN);
   error(udb==NULL,1,"alloc_ud [uflds.c]",
         "Unable to allocate memory space for the gauge field");
   set_ud2unity(4*VOLUME_TRD,2,udb);

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   #pragma omp target enter data map(to: udb[:n])
}

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
   error(udbv==NULL,1,"alloc_ud [uflds.c]",
         "Unable to allocate memory space for the gauge field");
   set_ud2unityv(4*VOLUME_TRD,2,udbv);

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   enter_su3_mat_field(udbv);
}


su3_dble *udfld(void)
{
   if (udb==NULL)
      alloc_ud();

   return udb;
}

su3_mat_field *udfldv(void)
{
   if (udbv==NULL)
      alloc_udv();

   return udbv;
}


void random_ud(void)
{
   int bc;
   int k,ix,mu,t;
   su3_dble *ud;

   if (udb==NULL)
      alloc_ud();

   bc=bc_type();

#pragma omp parallel private(k,ix,mu,t,ud)
   {
      k=omp_get_thread_num();

      for (mu=0;mu<4;mu++)
      {
         for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
         {
            t=global_time(ix);
            ud=udb+mu*VOLUME+ix;

            if (mu==0)
            {
               if ((t!=(N0-1))||(bc!=0))
                  random_su3_dble(ud);
            }
            else
            {
               if ((t!=0)||(bc!=1))
                  random_su3_dble(ud);
            }
         }
      }
   }

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   #pragma omp target update to(udb)
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

      for (mu=0;mu<4;mu++)
      {
         for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
         {
            t=global_time(ix);

            if (mu==0)
            {
               if ((t!=(N0-1))||(bc!=0))
                  random_su3_mat_field(udbv, ix);
            }
            else
            {
               if ((t!=0)||(bc!=1))
                  random_su3_mat_field(udbv, ix);
            }
         }
      }
   }

   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);
   set_bc();
   #pragma omp target update to(udbv)
}

static int set_phase(int pm,double *theta)
{
   int is;
   double p;

   p=theta[0]/(double)(N1);
   phase[0].re=cos(p);
   phase[0].im=sin(p);
   is=(p!=0.0);

   p=theta[1]/(double)(N2);
   phase[1].re=cos(p);
   phase[1].im=sin(p);
   is|=(p!=0.0);

   p=theta[2]/(double)(N3);
   phase[2].re=cos(p);
   phase[2].im=sin(p);
   is|=(p!=0.0);

   if (pm==-1)
   {
      phase[0].im=-phase[0].im;
      phase[1].im=-phase[1].im;
      phase[2].im=-phase[2].im;
   }

   return is;
}


static void mul_ud_phase(void)
{
   int k,l;
   su3_dble *ud,*um;

#pragma omp parallel private(k,l,ud,um)
   {
      k=omp_get_thread_num();

      for (l=0;l<3;l++)
      {
         ud=udb+(l+1)*VOLUME+k*VOLUME_TRD;
         um=ud+VOLUME_TRD;

         for (;ud<um;ud++)
            cm3x3_mulc(phase+l,ud,ud);
      }
   }
}


static void chs_ud0(void)
{
   int n,m;
   int nlks,*lks;
   int k,*lk,*lkm;
   double r;
   su3_dble *ud;

   lks=bnd_lks(&nlks);
   n=nlks/NTHREAD;
   m=nlks-n*NTHREAD;

#pragma omp parallel private(k,lk,lkm,r,ud)
   {
      k=omp_get_thread_num();

      lk=lks+k*n;
      lkm=lk+n;
      if (k==(NTHREAD-1))
         lkm+=m;
      r=-1.0;

      for (;lk<lkm;lk++)
      {
         ud=udb+(*lk);
         cm3x3_mulr(&r,ud,ud);
      }
   }
}


void set_ud_phase(void)
{
   int bc,is;
   su3_dble *ud;
   bc_parms_t bcp;

   if (query_flags(UD_PHASE_SET)==0)
   {
      if (udb==NULL)
         alloc_ud();

      bcp=bc_parms();
      bc=bcp.type;
      is=set_phase(1,bcp.theta);

      if (is)
      {
         mul_ud_phase();

         if ((cpr[0]==(NPROC0-1))&&((bc==1)||(bc==2)))
         {
            ud=udb+4*VOLUME+7*(BNDRY/4);
            cm3x3_mulc(phase,ud,ud);
            ud+=1;
            cm3x3_mulc(phase+1,ud,ud);
            ud+=1;
            cm3x3_mulc(phase+2,ud,ud);
         }
      }

      if (bc==3)
         chs_ud0();

      set_flags(UPDATED_UD);
      set_flags(SET_UD_PHASE);
   }
}


void unset_ud_phase(void)
{
   int bc,is;
   bc_parms_t bcp;

   if (query_flags(UD_PHASE_SET)==1)
   {
      set_flags(UNSET_UD_PHASE);

      if (udb==NULL)
         alloc_ud();

      bcp=bc_parms();
      bc=bcp.type;
      is=set_phase(-1,bcp.theta);

      if (is)
      {
         mul_ud_phase();

         if ((bc==1)||(bc==2))
            set_bc();
      }

      if (bc==3)
         chs_ud0();

      set_flags(UPDATED_UD);
   }
}


void renormalize_ud(void)
{
   int bc;
   int k,ix,mu,t;
   su3_dble *ud;

   if (query_flags(UD_PHASE_SET)==0)
   {
      if (udb==NULL)
         error_loc(1,1,"renormalize_ud [uflds.c]",
                   "Double-precision gauge field is not allocated");

      bc=bc_type();

#pragma omp parallel private(k,ix,mu,t,ud)
      {
         k=omp_get_thread_num();

         for (mu=0;mu<4;mu++)
         {
            for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
            {
               t=global_time(ix);
               ud=udb+mu*VOLUME+ix;

               if (mu==0)
               {
                  if ((t!=(N0-1))||(bc!=0))
                     project_to_su3_dble(ud);
               }
               else
               {
                  if ((t!=0)||(bc!=1))
                     project_to_su3_dble(ud);
               }
            }
         }
      }

      set_flags(UPDATED_UD);
   }
   else
      error_loc(1,1,"renormalize_ud [udflds.c]",
                "Attempt to renormalize non-unimodular link variables");
}


void assign_ud2u(void)
{
   int k,l,mu;
   float *r;
   double *rd;
   umat_t *u,*um;
   umat_dble_t *ud;

   if (ub==NULL)
      alloc_u();
   if (udb==NULL)
      error_loc(1,1,"assign_ud2u [uflds.c]",
                "Double-precision gauge field is not allocated");

#pragma omp parallel private(k,l,mu,u,um,ud,r,rd)
   {
      k=omp_get_thread_num();

      for (mu=0;mu<4;mu++)
      {
         u=(umat_t*)(ub+mu*VOLUME+k*VOLUME_TRD);
         um=u+VOLUME_TRD;
         ud=(umat_dble_t*)(udb+mu*VOLUME+k*VOLUME_TRD);

         for (;u<um;u++)
         {
            r=(*u).r;
            rd=(*ud).r;

            for (l=0;l<18;l++)
               r[l]=(float)(rd[l]);

            ud+=1;
         }
      }
   }

   set_flags(ASSIGNED_UD2U);
}
