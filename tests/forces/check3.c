/*******************************************************************************
*
* File check3.c
*
* Unit tests for force0() and action0().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "forces.h"
#include "devfcts.h"
#include "global.h"
#include "minitest.h"

#define N0 (NPROC0*L0)

static double c_g=0.789;


static qflt dSdt(double c)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   check_active((*mdfs).mom);

   // force0 runs on the GPU
   force0(c);
   check_active((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME_TRD,3,(*mdfs).mom,(*mdfs).frc);
}


static double chk_chs(double c)
{
   double dev;
   qflt rqsm;
   su3_alg_dble **wfd;
   mdflds_t *mdfs;

   wfd=reserve_wfd(1);
   mdfs=mdflds();

   random_ud_reproducible();
   force0(c);
   assign_alg2alg(4*VOLUME_TRD,2,(*mdfs).frc,wfd[0]);

   set_ud_phase();
   force0(c);
   muladd_assign_alg(4*VOLUME_TRD,2,-1.0,(*mdfs).frc,wfd[0]);
   rqsm=norm_square_alg(4*VOLUME_TRD,3,wfd[0]);
   dev=rqsm.q[0];
   rqsm=norm_square_alg(4*VOLUME_TRD,3,(*mdfs).frc);
   dev/=rqsm.q[0];
   release_wfd();

   return sqrt(dev);
}

TEST(Force0, NormSquareForce)
{
   qflt nrm_sq;
   mdflds_t *mdfs;
   
   mdfs=mdflds();
   
   random_ud_reproducible();
   force0(c_g);
   check_active((*mdfs).frc);
   nrm_sq = norm_square_alg(4*VOLUME_TRD,3,(*mdfs).frc);

   MT_PRINT("norm square frc: %f", fabs(nrm_sq.q[0]));
   EXPECT_TRUE(fabs(nrm_sq.q[0])>0);
}


TEST(Force0, PhaseInvariance)
{
   double dev=chk_chs(c_g);
   EXPECT_NEAR(dev, 0.0, 1.0e-12);
}


TEST(Force0, ForceVsActionDerivative)
{
   int k,ie;
   double eps,dev_frc,sig_loss;
   qflt dsdt,act,act0,act1;

   // action0 runs on the CPU
   for (k=0;k<4;k++)
   {
      random_ud_reproducible();
      set_ud_phase();
      random_mom();
      dsdt=dSdt(c_g);

      eps=2.0e-4;
      rot_ud(eps);
      act0=action0(1);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action0(1);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action0(1);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action0(1);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      scl_qflt(c_g,act0.q);
      scl_qflt(c_g,act1.q);

      add_qflt(act0.q,act1.q,act.q);
      sig_loss=-log10(fabs(act.q[0]/act0.q[0]));

      scl_qflt(-1.0/eps,act.q);
      MT_PRINT("dsdt.q: %f, act.q: %f", dsdt.q[0], act.q[0]);
      add_qflt(dsdt.q,act.q,act.q);
      dev_frc=act.q[0]/dsdt.q[0];

      unset_ud_phase();
      ie=check_bc(0.0);

      EXPECT_EQ(ie, 1);
      // machine precission minus significant loss as an approximate of finite
      // differences and volume
      EXPECT_NEAR(fabs(dev_frc), 0.0, pow(10.0, -(15.0-sig_loss)));
   }
}


static mt_test_t tests[] = {
   MT_TEST(Force0, NormSquareForce),
   MT_TEST(Force0, PhaseInvariance),
   MT_TEST(Force0, ForceVsActionDerivative),
};


int main(int argc,char *argv[])
{
   int my_rank,bc,iact;
   double phi[2],phi_prime[2],theta[3];

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");
      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(3.5,0.33,0,NULL,0,1.0);
   print_lat_parms(0x1);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;

   iact=0;
   set_hmc_parms(1,&iact,0,0,NULL,1,1.0);
   set_bc_parms(bc,0.9012,1.2034,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x3);

   start_ranlux(0,1234);
   geometry();
   alloc_wfd(1);

   int result=RUN_ALL_TESTS(my_rank,tests);

   MPI_Finalize();
   return result;
}
