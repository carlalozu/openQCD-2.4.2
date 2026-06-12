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
   SKIP_TEST("set_ud_phase() is not available in lexicographical data layout");
}


TEST(Force0, ForceVsActionDerivative)
{
   SKIP_TEST("rot_ud() is not available in lexicographical data layout");
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
