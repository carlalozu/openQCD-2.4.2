/*******************************************************************************
 *
 * File time.c
 *
 * Copyright (C) 2005-2016, 2018, 2022 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check of the programs for the plaquette sums of the double-precision
 * gauge field.
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "devfcts.h"
#include "uflds.h"
#include "global.h"
#include "profiler.h"

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

#define WARMUP_ITERS  3
#define PROFILE_ITERS 20

int main(int argc, char *argv[])
{
   prof_section s_kernel  = {.name = "plaq_sum_dble"};
   prof_section s_total   = {.name = "total"};

   int my_rank, bc;
   double phi[2], phi_prime[2], theta[3];
   double nplaq1, nplaq2, p1, p2;
   double d1, d2;
   FILE *flog = NULL;

   mpi_init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   prof_begin(&s_total);

   if (my_rank == 0)
   {
      flog = freopen("time_plaq_sum_dble.log", "w", stdout);

      printf("\n");
      printf("Plaquette sums of the double-precision gauge field\n");
      printf("--------------------------------------------------\n\n");

      print_lattice_sizes();

      bc = find_opt(argc, argv, "-bc");

      if (bc != 0)
         error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [time.c]",
                    "Syntax: time [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);
   phi[0] = 0.123;
   phi[1] = -0.534;
   phi_prime[0] = 0.912;
   phi_prime[1] = 0.078;
   theta[0] = 0.0;
   theta[1] = 0.0;
   theta[2] = 0.0;
   set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
   print_bc_parms(0x0);

   start_ranlux(0, 12345);
   geometry();

   /* -------------------------------------------------------------------------
    * Correctness check on freshly initialised (unit) field
    * ---------------------------------------------------------------------- */
   p1 = plaq_sum_dble(1);
   p2 = plaq_wsum_dble(1);

   if (bc == 0)
   {
      nplaq1 = (double)((6 * N0 - 3) * N1) * (double)(N2 * N3);
      nplaq2 = (double)((6 * N0 - 6) * N1) * (double)(N2 * N3);
   }
   else if (bc == 3)
   {
      nplaq1 = (double)(6 * N0 * N1) * (double)(N2 * N3);
      nplaq2 = nplaq1;
   }
   else
   {
      nplaq1 = (double)((6 * N0 + 3) * N1) * (double)(N2 * N3);
      nplaq2 = (double)(6 * N0 * N1) * (double)(N2 * N3);
   }

   d1 = d2 = 0.0;

   if (bc == 1)
   {
      d1 = cos(phi[0] / (double)(N1)) +
           cos(phi[1] / (double)(N1)) +
           cos((phi[0] + phi[1]) / (double)(N1)) +
           cos(phi[0] / (double)(N2)) +
           cos(phi[1] / (double)(N2)) +
           cos((phi[0] + phi[1]) / (double)(N2)) +
           cos(phi[0] / (double)(N3)) +
           cos(phi[1] / (double)(N3)) +
           cos((phi[0] + phi[1]) / (double)(N3));

      d1 = (d1 - 9.0) * (double)(N1 * N2 * N3);
   }

   if ((bc == 1) || (bc == 2))
   {
      d2 = cos(phi_prime[0] / (double)(N1)) +
           cos(phi_prime[1] / (double)(N1)) +
           cos((phi_prime[0] + phi_prime[1]) / (double)(N1)) +
           cos(phi_prime[0] / (double)(N2)) +
           cos(phi_prime[1] / (double)(N2)) +
           cos((phi_prime[0] + phi_prime[1]) / (double)(N2)) +
           cos(phi_prime[0] / (double)(N3)) +
           cos(phi_prime[1] / (double)(N3)) +
           cos((phi_prime[0] + phi_prime[1]) / (double)(N3));

      d2 = (d2 - 9.0) * (double)(N1 * N2 * N3);
   }

   if (my_rank == 0)
   {
      printf("After field initialization:\n");
      printf("Deviation from expected value (plaq_sum)  = %.1e\n",
             fabs(1.0 - p1 / (3.0 * nplaq1 + d1 + d2)));
      printf("Deviation from expected value (plaq_wsum) = %.1e\n\n",
             fabs(1.0 - p2 / (3.0 * nplaq2 + d1 + d2)));
   }

   /* -------------------------------------------------------------------------
    * Warmup: randomise field and call plaq_sum_dble without recording.
    * ---------------------------------------------------------------------- */
   if (my_rank == 0)
      printf("Running %d warmup iterations...\n", WARMUP_ITERS);

   for (int count = 0; count < WARMUP_ITERS; count++)
   {
      random_ud();
      (void)plaq_sum_dble(1);
   }

   if (my_rank == 0)
      printf("Warmup done. Starting timed benchmark...\n\n");

   /* -------------------------------------------------------------------------
    * Timed benchmark: PROFILE_ITERS iterations, each with a fresh random field.
    * ---------------------------------------------------------------------- */
   double result = 0.0;

   for (int count = 0; count < PROFILE_ITERS; count++)
   {
      random_ud();

      prof_begin(&s_kernel);
      result = plaq_sum_dble(1);
      prof_end(&s_kernel);
   }

   prof_end(&s_total);

   if (my_rank == 0)
   {
      int    flops    = 432 * 6 * VOLUME;
      double avg_time = s_kernel.total / (double)s_kernel.count;

      printf("Local size of the gauge field (KB): %d\n",
             (int)(72 * VOLUME * sizeof(double) / 1024));
      printf("VOLUME: %d   VOLUME_TRD: %d\n", VOLUME, VOLUME_TRD);
      printf("Measurement iterations: %lld\n\n", (long long)s_kernel.count);

      printf("avg kernel time              : %.9f s\n", avg_time);
      printf("Total GFlops/s               : %.4f\n",
             (double)flops * 1e-9 / avg_time);
      printf("GFlops/s per thread          : %.4f\n",
             (double)flops * 1e-9 / avg_time / (double)VOLUME_TRD);
      printf("Last result                  : %f\n\n", result);

      prof_report(&s_kernel);
      prof_report(&s_total);
   }

   if (my_rank == 0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}