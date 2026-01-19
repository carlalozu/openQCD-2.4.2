/*******************************************************************************
*
* File time1.c
*
* This file is heavily inspired by devel/dirac/time1.c by Martin Luescher
* 
* Timing of plaq_sum_dble(int n,int ix).
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "linalg.h"
#include "global.h"


int main(int argc,char *argv[])
{

   double plaq_time = 0.0;
   int my_rank,bc,count,nt,n_iter;
   double sm;
   double phi[2],phi_prime[2],theta[3];
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6_mpi1_openmp12_16x16x16x16_test_107.log","w",stdout);

      printf("\n");
      printf("Verify output of plaq_dble()\n");
      printf("--------------------------\n\n");

      print_lattice_sizes();

      if ((VOLUME*sizeof(float))<(64*1024))
      {
         printf("The local size of the gauge field is %d KB\n",
                (int)((72*VOLUME*sizeof(float))/(1024)));
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(float))/(1024)));
      }
      else
      {
         printf("The local size of the gauge field is %d MB\n",
                (int)((72*VOLUME*sizeof(float))/(1024*1024)));
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(float))/(1024*1024)));
      }

#if (defined x64)
#if (defined AVX)
#if (defined FMA3)
   printf("Using AVX and FMA3 instructions\n");
#else
   printf("Using AVX instructions\n");
#endif
#else
      printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
      printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
      printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
      printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
      printf("SSE prefetch instructions are not used\n");
#endif
#endif
      printf("\n");

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [time1.c]",
                    "Syntax: time1 [-bc <type>]");
      fflush(flog); 

   }


   set_lat_parms(5.5,1.0,0,NULL,0,1.978);
   print_lat_parms(0x2);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.35;
   theta[1]=-1.25;
   theta[2]=0.78;

   set_bc_parms(bc,0.55,0.78,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(2);

   start_ranlux(0,12345);
   geometry();

   nt=(int)(1.0e6f/(double)(VOLUME_TRD));
   if (nt<2)
      nt=2;

   n_iter=0; 

   do
   {
      for (count=0;count<nt;count++)
      {
         random_ud();
         MPI_Barrier(MPI_COMM_WORLD);
         
         sm = plaq_sum_dble(1);

	      n_iter += 1; 

         MPI_Barrier(MPI_COMM_WORLD);
          
      }
      
      nt*=2;
   } while (n_iter<2);

   if (my_rank==0)
   {
      printf("Result of plaq_sum_dble:\n");
      printf("%f \n\n",sm);
      printf("Number of iterations: ");
      printf("%d \n\n",n_iter);  
      printf("Time call to plaq_sum_dble:\n");
      printf("%f micro sec\n\n",plaq_time);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
