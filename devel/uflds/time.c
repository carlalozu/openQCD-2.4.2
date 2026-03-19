
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

void flush_cache(size_t flush_size, double* flush_buf)
{
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < flush_size; j++) {
        flush_buf[j] += 1.0; 
    }
}

int main(int argc, char *argv[])
{
   prof_section init_program = {.name = "init_program"};
   prof_section set_params = {.name = "set_params"};
   prof_section benchmark = {.name = "benchmark"};
   prof_section total = {.name = "total"};
   prof_section prepare_data = {.name = "prepare_data"};
   prof_section compute = {.name = "compute"};

   int my_rank, bc, nt, count;
   double phi[2], phi_prime[2], theta[3];
   double nplaq1, nplaq2, p1, p2;
   double d1, d2;
   double wt0, wt1, wt2, wdt, wdti;
   FILE *flog = NULL;
   
   mpi_init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
   prof_begin(&total);
   prof_begin(&init_program);
   if (my_rank == 0)
   {
      flog = freopen("time.log", "w", stdout);

      printf("\n");
      printf("Plaquette sums of the double-precision gauge field\n");
      printf("--------------------------------------------------\n\n");

      print_lattice_sizes();

      bc = find_opt(argc, argv, "-bc");

      if (bc != 0)
         error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [time.c]",
                    "Syntax: time [-bc <type>]");
   }
   prof_end(&init_program);

   prof_begin(&set_params);
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

   d1 = 0.0;
   d2 = 0.0;

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

   nt = (int)(1.0e6 / (double)(VOLUME));
   if (nt < 2)
      nt = 2;
   
   size_t flush_size = 114 * 4 * 1024 * 1024 / sizeof(double);
   double *flush_buf = malloc(flush_size * sizeof(double));
   
   flush_cache(flush_size, flush_buf);
   random_udv();
   prof_end(&set_params);
   
   prof_begin(&benchmark);
   wdti = 0.0;
   while (wdti < 5.0)
   {
      p1 = 0.0;
      wdt = 0.0;
      for (count = 0; count < nt; count++)
      {
         MPI_Barrier(MPI_COMM_WORLD);
         prof_begin(&prepare_data);
         wt0 = MPI_Wtime();
         flush_cache(flush_size, flush_buf);
         prof_end(&prepare_data);
         
         MPI_Barrier(MPI_COMM_WORLD);
         prof_begin(&compute);
         wt1 = MPI_Wtime();
         p1 += plaq_sum_dble(1);
         MPI_Barrier(MPI_COMM_WORLD);
         wt2 = MPI_Wtime();
         prof_end(&compute);

         wdt += wt2 - wt1;
         wdti += wt2 - wt0;
      }

      nt *= 2;
   }

   wdt = 2.0 * wdt / ((double)(nt));
   p1 = 2.0 * p1 / ((double)(nt));
   prof_end(&benchmark);
   prof_end(&total);

   if (my_rank == 0)
   {
      int flops = 432.0 * 6 * VOLUME;
      printf("Local size of the gauge field (KB): %d\n", (int)((72 * VOLUME * sizeof(double)) / (1024)));
      printf("Volume: %i\n", VOLUME);
      printf("Volume per thread: %i\n", VOLUME_TRD);
      printf("Block size: %i\n", BLOCK_SIZE);
      printf("Number of repetitions: %i\n", nt / 2);
      printf("Average time for plaq_sum_dble (sec): %.9f\n", wdt);
      printf("Flops: %d\n", flops); 
      printf("Total performance for plaq_sum_dble (MFlops/s): %d\n", (int)(flops * 1e-6 / wdt)); 
      printf("Time per lattice point & thread for plaq_sum_dble (sec): %.9f\n", wdt/((double)(VOLUME_TRD)));
      printf("Performance per thread for plaq_sum_dble (MFlops/s): %d\n", (int)(flops * 1e-6 / wdt));
      printf("Result: %f\n\n", p1);

      prof_report(&init_program);
      prof_report(&set_params);
      prof_report(&benchmark);
      prof_report(&prepare_data);
      prof_report(&compute);
      prof_report(&total);
   }

   if (my_rank == 0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
