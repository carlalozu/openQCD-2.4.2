/*******************************************************************************
*
* File check5.c
*
* Cross-check: GPU plaq_sum_dble and plaq_wsum_dble against a serial
* reference computed directly from the link variables.
*
* The test uses periodic boundary conditions (bc=3) so the reference loop
* is simply all 6 plaquettes at every site with no boundary weighting.
* The tolerance is loose enough to absorb non-associative GPU reductions.
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
#include "global.h"
#include "minitest.h"

#define TOL 1.0e-8


static double serial_plaq_sum_ref(void)
{
   int ix,n,ip[4];
   double pa=0.0,sm;
   su3_dble wd1,wd2,*udb;

   set_uidx();
   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   udb=udfld();

   for (ix=0;ix<VOLUME;ix++)
      for (n=0;n<6;n++)
      {
         plaq_uidx(n,ix,ip);
         su3xsu3(udb+ip[0],udb+ip[1],&wd1);
         su3dagxsu3dag(udb+ip[3],udb+ip[2],&wd2);
         cm3x3_retr(&wd1,&wd2,&sm);
         pa+=sm;
      }

   if (NPROC>1)
      MPI_Allreduce(MPI_IN_PLACE,&pa,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

   return pa;
}


TEST(CpuVsGpu, PlaqSum)
{
   int n;
   double p_gpu,p_ref;

   for (n=0;n<4;n++)
   {
      random_ud();
      p_gpu=plaq_sum_dble(1);
      p_ref=serial_plaq_sum_ref();
      EXPECT_NEAR(p_gpu, p_ref, TOL*fabs(p_ref)+1.0e-14);
   }
}


TEST(CpuVsGpu, PlaqWsum)
{
   int n;
   double p_wsum,p_ref;

   /* for bc==3, plaq_wsum_dble == plaq_sum_dble (no boundary weighting) */
   for (n=0;n<4;n++)
   {
      random_ud();
      p_wsum=plaq_wsum_dble(1);
      p_ref=serial_plaq_sum_ref();
      EXPECT_NEAR(p_wsum, p_ref, TOL*fabs(p_ref)+1.0e-14);
   }
}


static mt_test_t tests[] = {
   MT_TEST(CpuVsGpu, PlaqSum),
   MT_TEST(CpuVsGpu, PlaqWsum),
};


int main(int argc,char *argv[])
{
   int my_rank;
   double phi[2],phi_prime[2],theta[3];
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      print_lattice_sizes();
   }

   check_machine();

   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(3,1.0,1.0,1.0,1.0,phi,phi_prime,theta);

   start_ranlux(0,12345);
   geometry();

   int result=RUN_ALL_TESTS(my_rank,tests);

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   return result;
}
