
/*******************************************************************************
*
* File force0.c
*
* Copyright (C) 2005-2018, 2021, 2023 Martin Luescher, John Bulava
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Action of the double-precision gauge field and associated force.
*
*   void plaq_frc(void)
*     Computes the force deriving from the Wilson plaquette action,
*     omitting the prefactor 1/g0^2, and assigns the result to the MD
*     force field. In the case of open, SF or open-SF boundary conditions,
*     the boundary improvement coefficients are set to their tree-level
*     value independently of the values stored in the parameter data base.
*
*   void force0(double c)
*     Computes the force deriving from the gauge action, including the
*     prefactor 1/g0^2, multiplies the calculated force by c and assigns
*     the result to the MD force field. The coupling g0 and the other
*     parameters of the gauge action are retrieved from the parameter
*     data base.
*
*   qflt action0(int icom)
*     Computes the local part of the gauge action including the prefactor
*     1/g0^2. The coupling g0 and the other parameters of the action are
*     retrieved from the parameter data base. The program returns the sum
*     of the local parts of the action over all MPI processes if icom%2=1
*     and otherwise just the local part.
*
* The quadruple-precision types qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* See the notes doc/gauge_action.pdf for the definition of the gauge action
* and a description of the computation of the force deriving from it. The
* molecular-dynamics (MD) force field is the one returned by the program
* mdflds() (see mdflds/mdflds.c).
*
* On the links in the local lattice where the static link variables reside,
* the programs plaq_frc() and force0() set the force field to zero.
*
* The programs in this module assume that the simulation parameters have
* been entered to the parameter data base and that the geometry arrays are
* set. They perform global operations and are assumed to be called by the
* OpenMP master thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define FORCE0_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "forces.h"
#include "global.h"
#include "profiler.h"

#define N0 (NPROC0*L0)
#pragma omp declare target
static const int plns[6][2]={{0,1},{0,2},{0,3},{2,3},{3,1},{1,2}};
int nfc_fc0[8],ofs_fc0[8],hofs_fc0[8];
#pragma omp end declare target
static int init=0;
static su3_alg_dble *fdb;
static su3_dble *udb,*hdb;
prof_section force0_part_p = {.name = "force0_part", .level=2};

static void set_ofs(void)
{
   nfc_fc0[0]=FACE0/2;
   nfc_fc0[1]=FACE0/2;
   nfc_fc0[2]=FACE1/2;
   nfc_fc0[3]=FACE1/2;
   nfc_fc0[4]=FACE2/2;
   nfc_fc0[5]=FACE2/2;
   nfc_fc0[6]=FACE3/2;
   nfc_fc0[7]=FACE3/2;

   ofs_fc0[0]=VOLUME;
   ofs_fc0[1]=ofs_fc0[0]+(FACE0/2);
   ofs_fc0[2]=ofs_fc0[1]+(FACE0/2);
   ofs_fc0[3]=ofs_fc0[2]+(FACE1/2);
   ofs_fc0[4]=ofs_fc0[3]+(FACE1/2);
   ofs_fc0[5]=ofs_fc0[4]+(FACE2/2);
   ofs_fc0[6]=ofs_fc0[5]+(FACE2/2);
   ofs_fc0[7]=ofs_fc0[6]+(FACE3/2);

   hofs_fc0[0]=0;
   hofs_fc0[1]=hofs_fc0[0]+3*FACE0;
   hofs_fc0[2]=hofs_fc0[1]+3*FACE0;
   hofs_fc0[3]=hofs_fc0[2]+3*FACE1;
   hofs_fc0[4]=hofs_fc0[3]+3*FACE1;
   hofs_fc0[5]=hofs_fc0[4]+3*FACE2;
   hofs_fc0[6]=hofs_fc0[5]+3*FACE2;
   hofs_fc0[7]=hofs_fc0[6]+3*FACE3;

   init=1;
}


static void set_frc2zero_gpu(void)
{
   su3_alg_dble *frc;
   mdflds_t *mdfs;
   mdfs=mdflds();
   frc=(*mdfs).frc;

   #pragma omp target teams distribute parallel for
   for (int i=0; i<4*VOLUME; i++)
   {
      frc[i].c1 = 0.0;
      frc[i].c2 = 0.0;
      frc[i].c3 = 0.0;
      frc[i].c4 = 0.0;
      frc[i].c5 = 0.0;
      frc[i].c6 = 0.0;
      frc[i].c7 = 0.0;
      frc[i].c8 = 0.0;
   }

}


#pragma omp declare target
void set_staples(int n,int ix,int ia,su3_dble *vd,int (*idn)[4],int (*iup)[4],su3_dble *hdb,su3_dble *udb)
{
   int mu,nu,ifc;
   int iy,ib,ip[4];
   su3_dble wd[3] ALIGNED16;

   mu=plns[n][0];
   nu=plns[n][1];

   if (!ia)
   {
      iy=idn[ix][nu];

      if (iy<VOLUME)
      {
         _plaq_uidx(n,iy,ip,iup);

         su3xsu3(udb+ip[0],udb+ip[1],wd+2);
         su3dagxsu3(udb+ip[2],wd+2,vd);
      }
      else
      {
         ifc=2*nu;

         if (iy<(VOLUME+(BNDRY/2)))
            ib=iy-ofs_fc0[ifc];
         else
            ib=iy-ofs_fc0[ifc]-(BNDRY/2)+nfc_fc0[ifc];

         vd[0]=hdb[hofs_fc0[ifc]+3*ib+mu-(mu>nu)];
      }
   }

   iy=iup[ix][mu];

   if (iy<VOLUME)
   {
      _plaq_uidx(n,iy,ip,iup);

      su3xsu3dag(udb+ip[1],udb+ip[3],wd+2);
      su3xsu3(udb+ip[0],wd+2,vd+1);
   }
   else
   {
      ifc=2*mu+1;

      if (iy<(VOLUME+(BNDRY/2)))
         ib=iy-ofs_fc0[ifc];
      else
         ib=iy-ofs_fc0[ifc]-(BNDRY/2)+nfc_fc0[ifc];

      vd[1]=hdb[hofs_fc0[ifc]+3*ib+nu-(nu>mu)];
   }

   if (!ia)
   {
      iy=idn[ix][mu];

      if (iy<VOLUME)
      {
         _plaq_uidx(n,iy,ip,iup);

         su3xsu3(udb+ip[2],udb+ip[3],wd+2);
         su3dagxsu3(udb+ip[0],wd+2,vd+2);
      }
      else
      {
         ifc=2*mu;

         if (iy<(VOLUME+(BNDRY/2)))
            ib=iy-ofs_fc0[ifc];
         else
            ib=iy-ofs_fc0[ifc]-(BNDRY/2)+nfc_fc0[ifc];

         vd[2]=hdb[hofs_fc0[ifc]+3*ib+nu-(nu>mu)];
      }
   }

   iy=iup[ix][nu];

   if (iy<VOLUME)
   {
      _plaq_uidx(n,iy,ip,iup);

      su3xsu3dag(udb+ip[3],udb+ip[1],wd+2);
      su3xsu3(udb+ip[2],wd+2,vd+3);
   }
   else
   {
      ifc=2*nu+1;

      if (iy<(VOLUME+(BNDRY/2)))
         ib=iy-ofs_fc0[ifc];
      else
         ib=iy-ofs_fc0[ifc]-(BNDRY/2)+nfc_fc0[ifc];

      vd[3]=hdb[hofs_fc0[ifc]+3*ib+mu-(mu>nu)];
   }
}
#pragma omp end declare target

static void plaq_frc_part(int ix,int bc,int (*iup)[4],su3_dble *udb,su3_alg_dble *fdb)
{
   int n,t,ip[4];
   double r;
   su3_alg_dble X ALIGNED16;
   su3_dble wd[2] ALIGNED16;

   // bc=bc_type();

   // for (ix=ofs_pt;ix<(ofs_pt+vol);ix++)
   {
      t=global_time(ix);

      if ((t<(N0-1))||(bc!=0))
      {
         for (n=0;n<3;n++)
         {
            _plaq_uidx(n,ix,ip,iup);

            su3xsu3dag(udb+ip[1],udb+ip[3],wd);
            su3dagxsu3(udb+ip[2],udb+ip[0],wd+1);

            if ((t<(N0-1))||(bc==3))
            {
               prod2su3alg(wd,wd+1,&X);
               _su3_alg_add_assign(*(fdb+ip[1]),X);
            }

            prod2su3alg(wd+1,wd,&X);
            _su3_alg_sub_assign(*(fdb+ip[3]),X);

            su3xsu3dag(wd,udb+ip[2],wd+1);
            prod2su3alg(udb+ip[0],wd+1,&X);
            _su3_alg_add_assign(*(fdb+ip[0]),X);

            if ((t>0)||(bc!=1))
            {
               _su3_alg_sub_assign(*(fdb+ip[2]),X);
            }
         }
      }

      if ((t>0)||(bc!=1))
      {
         r=1.0;

         if (((t==0)&&(bc!=3))||((t==(N0-1))&&(bc==0)))
            r=0.5;

         for (n=3;n<6;n++)
         {
            _plaq_uidx(n,ix,ip,iup);

            su3xsu3dag(udb+ip[1],udb+ip[3],wd);
            su3dagxsu3(udb+ip[2],udb+ip[0],wd+1);
            prod2su3alg(wd,wd+1,&X);
            _su3_alg_mul_add_assign(*(fdb+ip[1]),r,X);
            prod2su3alg(wd+1,wd,&X);
            _su3_alg_mul_sub_assign(*(fdb+ip[3]),r,X);

            su3xsu3dag(wd,udb+ip[2],wd+1);
            prod2su3alg(udb+ip[0],wd+1,&X);
            _su3_alg_mul_add_assign(*(fdb+ip[0]),r,X);
            _su3_alg_mul_sub_assign(*(fdb+ip[2]),r,X);
         }
      }
   }
}


void plaq_frc(void)
{
   int k,isb,ofs_pt,vol;
   mdflds_t *mdfs;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   set_uidx();
   udb=udfld();
   mdfs=mdflds();
   fdb=(*mdfs).frc;
   set_frc2zero_gpu();

   int bc=bc_type();
   #pragma omp target update to(udb[0:4*VOLUME], fdb[0:4*VOLUME])

   #pragma omp target teams distribute parallel for
   for (int ix=0; ix<VOLUME; ix++)
   {
      plaq_frc_part(ix,bc,iup,udb,fdb);
   }
   #pragma omp target update from(fdb[0:4*VOLUME])

   // this function is not implemented at the moment
   // add_bnd_frc();
}


static void force0_part(int ix,bc_parms_t bcp,lat_parms_t lat,double c,int (*iup)[4],int (*idn)[4],su3_dble *udb,su3_alg_dble *fdb,su3_dble *hdb)
{
   int bc,n,t,ip[4];
   double r0,r1,c0,c1,*cG;
   su3_alg_dble X ALIGNED16;
   su3_dble wd[3] ALIGNED16;
   su3_dble vd[4] ALIGNED16;
   // lat_parms_t lat;
   // bc_parms_t bcp;

   // lat=lat_parms();
   c*=(lat.beta/6.0);
   c0=lat.c0;
   c1=lat.c1;

   // bcp=bc_parms();
   bc=bcp.type;
   cG=bcp.cG;

   // for (ix=ofs_pt;ix<(ofs_pt+vol);ix++)
   {
      t=global_time(ix);

      if ((t<(N0-1))||(bc!=0))
      {
         r0=c*c0;
         r1=c*c1;

         if ((t==0)&&(bc==1))
            r0*=cG[0];
         else if ((t==(N0-1))&&(bc!=3))
            r0*=cG[1];

         for (n=0;n<3;n++)
         {
            _plaq_uidx(n,ix,ip,iup);

            su3xsu3dag(udb+ip[1],udb+ip[3],wd);
            su3dagxsu3(udb+ip[2],udb+ip[0],wd+1);

            if ((t<(N0-1))||(bc==3))
            {
               prod2su3alg(wd,wd+1,&X);
               su3_alg_mul_add_assign(fdb+ip[1],r0,X);
	    }

            prod2su3alg(wd+1,wd,&X);
            su3_alg_mul_sub_assign(fdb+ip[3],r0,X);

            su3xsu3dag(wd,udb+ip[2],wd+1);
            prod2su3alg(udb+ip[0],wd+1,&X);
            su3_alg_mul_add_assign(fdb+ip[0],r0,X);

            if ((t>0)||(bc!=1))
            {
               su3_alg_mul_sub_assign(fdb+ip[2],r0,X);
            }

            if (c0!=1.0)
            {
               set_staples(n,ix,0,vd,idn,iup,hdb,udb);

               if ((t==0)&&(bc==1))
               {
                  su3xsu3(wd+1,udb+ip[0],wd+2);
                  su3xsu3(udb+ip[0],wd+2,wd+2);

                  prod2su3alg(wd+1,wd+2,&X);
                  su3_alg_mul_add_assign(fdb+ip[1],r1,X);

                  prod2su3alg(wd+2,wd+1,&X);
                  su3_alg_mul_add_assign(fdb+ip[0],r1,X);

                  su3dagxsu3(udb+ip[2],wd+2,wd+2);

                  prod2su3alg(wd+2,wd,&X);
                  su3_alg_mul_sub_assign(fdb+ip[3],r1,X);
               }

               if ((t==(N0-1))&&(bc!=3))
               {
                  su3xsu3(wd+1,udb+ip[0],wd+2);
                  su3xsu3(udb+ip[0],wd+2,wd+2);

                  prod2su3alg(wd+2,wd+1,&X);
                  su3_alg_mul_add_assign(fdb+ip[0],r1,X);
                  su3_alg_mul_sub_assign(fdb+ip[2],r1,X);

                  su3dagxsu3(udb+ip[2],wd+2,wd+2);

                  prod2su3alg(wd+2,wd,&X);
                  su3_alg_mul_sub_assign(fdb+ip[3],r1,X);
               }

               if ((t<(N0-1))||(bc==3))
               {
                  prod2su3alg(wd+1,vd,&X);
                  su3_alg_mul_add_assign(fdb+ip[1],r1,X);
               }

               if ((t>0)||(bc!=1))
               {
                  prod2su3alg(vd,wd+1,&X);
                  su3_alg_mul_sub_assign(fdb+ip[2],r1,X);
               }

               su3dagxsu3(udb+ip[2],vd,wd+1);
               prod2su3alg(wd+1,wd,&X);
               su3_alg_mul_sub_assign(fdb+ip[3],r1,X);

               if ((t<(N0-2))||((t==(N0-2))&&(bc!=0))||(bc==3))
               {
                  su3xsu3dag(udb+ip[3],vd+1,wd+1);
                  su3xsu3dag(wd+1,udb+ip[0],wd+2);
                  prod2su3alg(udb+ip[2],wd+2,&X);
                  su3_alg_mul_sub_assign(fdb+ip[0],r1,X);

                  if ((t>0)||(bc!=1))
                  {
                     su3_alg_mul_add_assign(fdb+ip[2],r1,X);
                  }

                  prod2su3alg(wd+2,udb+ip[2],&X);
                  su3_alg_mul_add_assign(fdb+ip[3],r1,X);
               }

               if ((t>0)||(bc==3))
               {
                  su3xsu3dag(wd,vd+2,wd+1);
                  prod2su3alg(udb+ip[0],wd+1,&X);
                  su3_alg_mul_add_assign(fdb+ip[0],r1,X);

                  if ((t<(N0-1))||(bc==3))
                  {
                     prod2su3alg(wd+1,udb+ip[0],&X);
                     su3_alg_mul_add_assign(fdb+ip[1],r1,X);
                  }

                  su3dagxsu3(vd+2,udb+ip[0],wd+1);
                  prod2su3alg(wd+1,wd,&X);
                  su3_alg_mul_sub_assign(fdb+ip[3],r1,X);
               }

               su3xsu3dag(udb+ip[1],vd+3,wd);
               su3xsu3dag(wd,udb+ip[2],wd+1);
               prod2su3alg(udb+ip[0],wd+1,&X);
               su3_alg_mul_add_assign(fdb+ip[0],r1,X);

               if ((t>0)||(bc!=1))
               {
                  su3_alg_mul_sub_assign(fdb+ip[2],r1,X);
               }

               if ((t<(N0-1))||(bc==3))
               {
                  prod2su3alg(wd+1,udb+ip[0],&X);
                  su3_alg_mul_add_assign(fdb+ip[1],r1,X);
               }
            }
         }
      }

      if ((t>0)||(bc!=1))
      {
         r0=c*c0;
         r1=c*c1;

         if ((t==0)&&(bc!=3))
         {
            r0*=(0.5*cG[0]);
            r1*=(0.5*cG[0]);
         }
         else if ((t==(N0-1))&&(bc==0))
         {
            r0*=(0.5*cG[1]);
            r1*=(0.5*cG[1]);
         }

         for (n=3;n<6;n++)
         {
            _plaq_uidx(n,ix,ip,iup);

            su3xsu3dag(udb+ip[1],udb+ip[3],wd);
            su3dagxsu3(udb+ip[2],udb+ip[0],wd+1);
            prod2su3alg(wd,wd+1,&X);
            su3_alg_mul_add_assign(fdb+ip[1],r0,X);

            prod2su3alg(wd+1,wd,&X);
            su3_alg_mul_sub_assign(fdb+ip[3],r0,X);

            su3xsu3dag(wd,udb+ip[2],wd+1);
            prod2su3alg(udb+ip[0],wd+1,&X);
            su3_alg_mul_add_assign(fdb+ip[0],r0,X);
            su3_alg_mul_sub_assign(fdb+ip[2],r0,X);

            if (c0!=1.0)
            {
               set_staples(n,ix,0,vd,idn,iup,hdb,udb);

               prod2su3alg(wd+1,vd,&X);
               su3_alg_mul_add_assign(fdb+ip[1],r1,X);

               prod2su3alg(vd,wd+1,&X);
               su3_alg_mul_sub_assign(fdb+ip[2],r1,X);

               su3dagxsu3(udb+ip[2],vd,wd+1);
               prod2su3alg(wd+1,wd,&X);
               su3_alg_mul_sub_assign(fdb+ip[3],r1,X);

               su3xsu3dag(udb+ip[3],vd+1,wd+1);
               su3xsu3dag(wd+1,udb+ip[0],wd+2);
               prod2su3alg(udb+ip[2],wd+2,&X);
               su3_alg_mul_sub_assign(fdb+ip[0],r1,X);
               su3_alg_mul_add_assign(fdb+ip[2],r1,X);

               prod2su3alg(wd+2,udb+ip[2],&X);
               su3_alg_mul_add_assign(fdb+ip[3],r1,X);

               su3xsu3dag(wd,vd+2,wd+1);
               prod2su3alg(udb+ip[0],wd+1,&X);
               su3_alg_mul_add_assign(fdb+ip[0],r1,X);

               prod2su3alg(wd+1,udb+ip[0],&X);
               su3_alg_mul_add_assign(fdb+ip[1],r1,X);

               su3dagxsu3(vd+2,udb+ip[0],wd+1);
               prod2su3alg(wd+1,wd,&X);
               su3_alg_mul_sub_assign(fdb+ip[3],r1,X);

               su3xsu3dag(udb+ip[1],vd+3,wd);
               su3xsu3dag(wd,udb+ip[2],wd+1);
               prod2su3alg(udb+ip[0],wd+1,&X);
               su3_alg_mul_add_assign(fdb+ip[0],r1,X);
               su3_alg_mul_sub_assign(fdb+ip[2],r1,X);

               prod2su3alg(wd+1,udb+ip[0],&X);
               su3_alg_mul_add_assign(fdb+ip[1],r1,X);
            }
         }
      }
   }
}


void force0(double c)
{
   // int k,isb,ofs_pt,vol;
   lat_parms_t lat;
   mdflds_t *mdfs;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   set_uidx();
   udb=udfld();
   mdfs=mdflds();
   fdb=(*mdfs).frc;

   lat=lat_parms();

   if (lat.c0==1.0)
      hdb=NULL;
   else
   {
      if (init==0)
         set_ofs();

      if (query_flags(BSTAP_UP2DATE)!=1)
         set_bstap();
      hdb=bstap();
   }
   bc_parms_t bc=bc_parms();

   set_frc2zero_gpu();

   prof_begin(&force0_part_p);
   #pragma omp target teams distribute parallel for
   for (int ix=0; ix<VOLUME; ix++)
   {
      force0_part(ix,bc,lat,c,iup,idn,udb,fdb,hdb);
   }
   prof_end(&force0_part_p);

   // add_bnd_frc();
}

#pragma omp declare target
static void wloops(int n,int ix,int t,double c0,double *trU,int (*iup)[4],int (*idn)[4],int bc,su3_dble *udb,su3_dble *hdb)
{
   int ip[4];
   su3_dble wd[2] ALIGNED16;
   su3_dble vd[4] ALIGNED16;

   _plaq_uidx(n,ix,ip,iup);

   trU[0]=0.0;
   trU[1]=0.0;
   trU[2]=0.0;
   trU[3]=0.0;

   if ((n>=3)||(t<(N0-1))||(bc!=0))
   {
      su3dagxsu3(udb+ip[2],udb+ip[0],wd);
      su3xsu3dag(udb+ip[1],udb+ip[3],wd+1);
      cm3x3_retr(wd,wd+1,trU);
      trU[0]=3.0-trU[0];
   }

   if (c0!=1.0)
   {
      set_staples(n,ix,1,vd,idn,iup,hdb,udb);

      if ((n<3)&&(((t==0)&&(bc==1))||
                  ((t==(N0-1))&&((bc==1)||(bc==2)))))
      {
         su3xsu3(wd,wd+1,wd+1);
         cm3x3_retr(wd+1,wd+1,trU+3);
         trU[3]=3.0-trU[3];
      }

      if ((n>=3)||(t<(N0-1))||(bc!=0))
      {
         su3xsu3dag(udb+ip[1],vd+3,wd+1);
         cm3x3_retr(wd,wd+1,trU+1);
         trU[1]=3.0-trU[1];
      }

      if ((n>=3)||(t<(N0-2))||((t==(N0-2))&&(bc!=0))||(bc==3))
      {
         su3xsu3dag(vd+1,udb+ip[3],wd+1);
         cm3x3_retr(wd,wd+1,trU+2);
         trU[2]=3.0-trU[2];
      }
   }
}
#pragma omp end declare target

static qflt action0_part(int ofs_pt,int vol)
{
   int bc,ix,t,n;
   double act1,c0,c1,*cG;
   double r0,r1,trU[4];
   qflt act0;
   lat_parms_t lat;
   bc_parms_t bcp;

   lat=lat_parms();
   c0=lat.c0;
   c1=lat.c1;

   bcp=bc_parms();
   bc=bcp.type;
   cG=bcp.cG;

   act0.q[0]=0.0;
   act0.q[1]=0.0;

   for (ix=ofs_pt;ix<(vol+ofs_pt);ix++)
   {
      t=global_time(ix);
      act1=0.0;

      if ((t<(N0-1))||(bc!=0))
      {
         r0=c0;

         if ((t==0)&&(bc==1))
            r0*=cG[0];
         else if ((t==(N0-1))&&(bc!=3))
            r0*=cG[1];

         for (n=0;n<3;n++)
         {
            wloops(n,ix,t,c0,trU,iup,idn,bc,udb,hdb);
            act1+=(r0*trU[0]+c1*(trU[1]+trU[2]+0.5*trU[3]));
         }
      }

      if ((t>0)||(bc!=1))
      {
         r0=c0;
         r1=c1;

         if ((t==0)&&(bc!=3))
         {
            r0*=(0.5*cG[0]);
            r1*=(0.5*cG[0]);
         }
         else if ((t==(N0-1))&&(bc==0))
         {
            r0*=(0.5*cG[1]);
            r1*=(0.5*cG[1]);
         }

         for (n=3;n<6;n++)
         {
            wloops(n,ix,t,c0,trU,iup,idn,bc,udb,hdb);
            act1+=(r0*trU[0]+r1*(trU[1]+trU[2]));
         }
      }

      acc_qflt(act1,act0.q);
   }

   return act0;
}


qflt action0(int icom)
{
   int k;
   double *qact[1],pa;
   qflt act0;
   lat_parms_t lat;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   set_uidx();
   udb=udfld();
   lat=lat_parms();

   if (lat.c0==1.0)
      hdb=NULL;
   else
   {
      if (init==0)
         set_ofs();

      if (query_flags(BSTAP_UP2DATE)!=1)
         set_bstap();
      hdb=bstap();
   }

   act0.q[0]=0.0;
   act0.q[1]=0.0;
   pa=0.0;

#pragma omp parallel private(k) reduction(+:pa)
   {
      qflt loc_act0;
      k=omp_get_thread_num();
      loc_act0=action0_part(k*VOLUME_TRD,VOLUME_TRD);
      pa+=loc_act0.q[0];
   }
   acc_qflt(pa,act0.q);

   if ((NPROC>1)&&(icom&0x1))
   {
      qact[0]=act0.q;
      global_qsum(1,qact,qact);
   }

   scl_qflt(lat.beta/3.0,act0.q);

   return act0;
}
