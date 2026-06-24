
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
   prof_section s_prepare  = {.name = "prepare_data"};
   prof_section s_upload   = {.name = "upload ufldv"};
   prof_section s_kernel   = {.name = "plaq_dblev"};
   prof_section s_total    = {.name = "total"};

   int my_rank, bc, count;
   double phi[2], phi_prime[2], theta[3];
   static su3_mat_field *udbv;

   mpi_init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
   prof_begin(&s_total);
   if (my_rank == 0)
   {

      printf("\n");
      printf("Plaquette action (plaq_dble) of the double-precision gauge field SoA\n");
      printf("--------------------------------------------------------------\n\n");

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
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
   print_bc_parms(0x0);

   start_ranlux(0, 12345);
   geometry();
   udbv = udfldv();
   
   if (my_rank == 0)
      printf("Running %d warmup iterations...\n", WARMUP_ITERS);

   double pa_warm;
   for (count = 0; count < WARMUP_ITERS; count++)
   {
      random_udv();
      pa_warm = local_plaq_dblev(0);
      (void)pa_warm;
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (my_rank == 0)
      printf("Warmup done. Starting timed benchmark...\n\n");


   double pa = 0.0;

   for (count = 0; count < PROFILE_ITERS; count++)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      prof_begin(&s_prepare);
      random_udv();
      prof_end(&s_prepare);
      
      prof_begin(&s_upload);
      update_su3_mat_field();
      prof_end(&s_upload);

      for (int n = 0; n < 6; n++)
      {
         MPI_Barrier(MPI_COMM_WORLD);
         prof_begin(&s_kernel);
         pa = local_plaq_dblev(n);
         prof_end(&s_kernel);
      }

   }   

   prof_end(&s_total);
   
   if (my_rank == 0)
   {
      int flops = 432 * VOLUME;
      double avg_time = s_kernel.total / (double)s_kernel.count;

      printf("\nLocal size of the gauge field (KB): %d\n", (int)((72 * VOLUME * sizeof(double)) / (1024)));
      printf("Volume: %i\n", VOLUME);
      printf("Volume per thread: %i\n", VOLUME_TRD);
      printf("Number of repetitions for final time: %i\n", (int)s_kernel.count);
      printf("Average time for plaq_dble (sec): %.9f\n", avg_time);
      printf("Flops: %d\n", flops); 
      printf("Total performance for plaq_dble (GFlops/s): %f\n", (double)(flops * 1e-9 / avg_time)); 
      printf("Time per lattice point & thread for plaq_dble (sec): %.9f\n", avg_time/((double)(VOLUME_TRD)));
      printf("Result: %f\n\n", pa);

      prof_report(&s_prepare);
      prof_report(&s_upload);
      prof_report(&s_kernel);
      prof_report(&s_total);
   }

   MPI_Finalize();
   exit(0);
}
