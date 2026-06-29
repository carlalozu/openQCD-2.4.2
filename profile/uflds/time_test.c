
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

   int my_rank, bc, nt, count;
   double phi[2], phi_prime[2], theta[3];
   double p1 = 0.0;
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
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
   print_bc_parms(0x0);

   start_ranlux(0, 12345);
   geometry();
   size_t flush_size = 114 * 4 * 1024 * 1024 / sizeof(double);
   double *flush_buf = malloc(flush_size * sizeof(double));
   
   flush_cache(flush_size, flush_buf);
   random_ud();
   prof_end(&set_params);
   
   nt = 0;
   // prof_reset(&compute);
   prof_begin(&benchmark);
   for (count = 0; count < nt; count++)
   {
         p1 += plaq_sum_dble(1);
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
      printf("Number of repetitions: %i\n", nt / 2);
      printf("Average time for plaq_sum_dble (sec): %.9f\n", wdt);
      printf("Flops: %d\n", flops); 
      printf("Total performance for plaq_sum_dble (GFlops/s): %f\n", (double)(flops * 1e-9 / wdt));
      printf("Time per lattice point & thread for plaq_sum_dble (sec): %.9f\n", wdt/((double)(VOLUME_TRD)));
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
