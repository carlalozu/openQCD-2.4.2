/*******************************************************************************
*
* File check4.c
*
* Computes force0 and momentum_action on a unit gauge field with zero momentum
* and in a random gauge and momentum fields.
*
* Usage: check4 [-bc <type>]
*   bc=0 open, bc=1 SF, bc=2 open-SF, bc=3 periodic (default: 3)
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
#include "global.h"
#include "minitest.h"

#define N0 (NPROC0*L0)

static double c_g = 1.0;


TEST(UnitGauge, Force0NormIsZero)
{
   qflt nrm_sq;
   mdflds_t *mdfs;

   set_ud2unity(4*VOLUME_TRD, 2, udfld());

   mdfs = mdflds();
   force0(c_g);

   nrm_sq = norm_square_alg(4*VOLUME_TRD, 3, (*mdfs).frc);

   if (mt_rank_ == 0)
      MT_PRINT("||force0||^2 on unit gauge field = %.6e", nrm_sq.q[0]);

   /* For periodic BC the unit gauge is the trivial minimum of the Wilson
      action, so the force must vanish exactly. For open/SF boundary
      conditions a small non-zero boundary contribution is expected. */
   EXPECT_NEAR(nrm_sq.q[0], 0.0, 1.0e-10);
}


TEST(UnitGauge, MomentumActionZero)
{
   qflt act;

   mdflds_t *mdfs;

   mdfs = mdflds();
   set_alg2zero(4*VOLUME_TRD, 2, (*mdfs).mom);
   update_mom(); /* Accumulate frc into mom. */
   act = momentum_action(1);

   if (mt_rank_ == 0)
      MT_PRINT("momentum_action with zero momenta = %.15e", act.q[0]);

   EXPECT_NEAR(act.q[0], 0.0, 1.0e-14);
}

TEST(RandomGauge, MomentumActionNotZero)
{
   qflt act,nrm_sq;
   mdflds_t *mdfs;
   
   mdfs = mdflds();
   random_mom();
   random_ud();

   force0(c_g);
   update_mom();

   nrm_sq = norm_square_alg(4*VOLUME_TRD, 3, (*mdfs).frc);
   act = momentum_action(1);

   if (mt_rank_ == 0)
      MT_PRINT("||force0||^2 on unit gauge field = %.6e", nrm_sq.q[0]);

   /* For periodic BC the unit gauge is the trivial minimum of the Wilson
      action, so the force must vanish exactly. For open/SF boundary
      conditions a small non-zero boundary contribution is expected. */
   EXPECT_TRUE(nrm_sq.q[0] > 0.0);

   if (mt_rank_ == 0)
      MT_PRINT("momentum_action with zero momenta = %.15e", act.q[0]);

   /* The momentum action is a sum of squared norms, so it must be positive. */
   EXPECT_TRUE(act.q[0] > 0.0);
}


static mt_test_t tests[] = {
   MT_TEST(UnitGauge, Force0NormIsZero),
   MT_TEST(UnitGauge, MomentumActionZero),
   MT_TEST(RandomGauge, MomentumActionNotZero),
};


int main(int argc, char *argv[])
{
   int my_rank, bc, iact;
   double phi[2], phi_prime[2], theta[3];

   mpi_init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   if (my_rank == 0)
   {
      print_lattice_sizes();

      bc = find_opt(argc, argv, "-bc");
      if (bc != 0)
         error_root(sscanf(argv[bc+1], "%d", &bc) != 1, 1,
                    "main [check4.c]", "Syntax: check4 [-bc <type>]");
      else
         bc = 3; /* default: periodic */
   }

   check_machine();

   /* beta=6.0, c0=1.0 (pure Wilson), no csw, no twisted mass */
   set_lat_parms(6.0, 1.0, 0, NULL, 0, 1.0);
   print_lat_parms(0x1);

   MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);

   phi[0]      =  0.0;   phi[1]       = 0.0;
   phi_prime[0]=  0.0;   phi_prime[1] = 0.0;
   theta[0]    =  0.0;   theta[1]     = 0.0;   theta[2] = 0.0;

   iact = 0;
   set_hmc_parms(1, &iact, 0, 0, NULL, 1, 1.0);
   set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
   print_bc_parms(0x3);

   start_ranlux(0, 1234);
   geometry();

   int result = RUN_ALL_TESTS(my_rank, tests);

   MPI_Finalize();
   return result;
}
