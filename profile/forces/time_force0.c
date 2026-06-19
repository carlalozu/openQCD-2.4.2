
/*******************************************************************************
 *
 * File time_force0.c
 *
 * Profiling of force0() - the plaquette gauge force computation.
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
#include "forces.h"
#include "linalg.h"
#include "global.h"
#include "profiler.h"

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

#define WARMUP_ITERS  3
#define PROFILE_ITERS 120

int main(int argc, char *argv[])
{
   prof_section s_prepare = {.name = "prepare_data"};
   prof_section s_kernel  = {.name = "force0"};
   prof_section s_total   = {.name = "total"};

   int my_rank, bc, iact;
   double phi[2], phi_prime[2], theta[3];
   qflt rqsm;
   mdflds_t *mdfs;

   mpi_init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   prof_begin(&s_total);

   if (my_rank == 0)
   {
      printf("\n");
      printf("Gauge force (force0) of the double-precision gauge field\n");
      printf("--------------------------------------------------------\n\n");

      print_lattice_sizes();

      bc = find_opt(argc, argv, "-bc");

      if (bc != 0)
         error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [time_force0.c]",
                    "Syntax: time_force0 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);

   set_lat_parms(5.5, 1.0, 0, NULL, 0, 1.0);
   print_lat_parms(0x2);

   phi[0] = 0.123;
   phi[1] = -0.534;
   phi_prime[0] = 0.912;
   phi_prime[1] = 0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;

   iact = 0;
   set_hmc_parms(1, &iact, 0, 0, NULL, 1, 1.0);
   set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
   print_bc_parms(0x3);

   start_ranlux(0, 12345);
   geometry();
   mdfs=mdflds();

   /* -------------------------------------------------------------------------
    * Warmup: randomise field and call force0 without recording.
    * ---------------------------------------------------------------------- */
   if (my_rank == 0)
      printf("Running %d warmup iterations...\n", WARMUP_ITERS);

   for (int count = 0; count < WARMUP_ITERS; count++)
   {
      random_ud_reproducible();
      force0(1.0);
   }

   if (my_rank == 0)
      printf("Warmup done. Starting timed benchmark...\n\n");

   /* -------------------------------------------------------------------------
    * Timed benchmark: PROFILE_ITERS iterations, each with a fresh random field.
    * ---------------------------------------------------------------------- */
   prof_reset(&force0_part_p);
   prof_reset(&update_force0_p);
   for (int count = 0; count < PROFILE_ITERS; count++)
   {
      prof_begin(&s_prepare);
      random_ud_reproducible();
      prof_end(&s_prepare);

      prof_begin(&s_kernel);
      force0(1.0);
      prof_end(&s_kernel);

   }
   
   rqsm=norm_square_alg(4*VOLUME_TRD,3,(*mdfs).frc);
   prof_end(&s_total);

   if (my_rank == 0)
   {
      /* 6 planes × (3 su3prod@198 + 3 prod2su3alg@216 + 4 alg_mul@16) per site, c0=1.0 bulk */
      long long flops = 7836LL * VOLUME;
      double avg_time = force0_part_p.total / (double)force0_part_p.count;

      printf("\nLocal size of the gauge field (KB): %d\n", (int)((72 * VOLUME * sizeof(double)) / 1024));
      printf("Local size of the force field  (KB): %d\n", (int)((64 * VOLUME * sizeof(double)) / 1024));
      printf("Volume: %i\n", VOLUME);
      printf("Volume per thread: %i\n", VOLUME_TRD);
      printf("Number of repetitions for final time: %i\n", (int)s_kernel.count);
      printf("Average time for force0 (sec): %.9f\n", avg_time);
      printf("Flops: %lld\n", flops);
      printf("Total performance for force0 (GFlops/s): %f\n", (double)(flops * 1e-9 / avg_time));
      printf("Time per lattice point & thread for force0 (sec): %.9f\n",
             avg_time / (double)VOLUME_TRD);
      printf("Result: %f\n\n", rqsm.q[0]/(4*VOLUME_TRD));

      prof_report(&s_prepare);
      prof_report(&s_kernel);
      prof_report(&force0_part_p);
      prof_report(&update_force0_p);
      prof_report(&s_total);
   }

   MPI_Finalize();
   exit(0);
}
