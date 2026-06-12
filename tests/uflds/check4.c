/*******************************************************************************
*
* File check4.c
*
* Unit tests for plaq_sum_dble, plaq_wsum_dble, and plaq_action_slices.
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
#include "devfcts.h"
#include "uflds.h"
#include "global.h"
#include "minitest.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int    bc_g;
static double phi_g[2],phi_prime_g[2];
static double nplaq1_g,nplaq2_g,d1_g,d2_g;


static void compute_expected(void)
{
   if (bc_g==0)
   {
      nplaq1_g=(double)((6*N0-3)*N1)*(double)(N2*N3);
      nplaq2_g=(double)((6*N0-6)*N1)*(double)(N2*N3);
   }
   else if (bc_g==3)
   {
      nplaq1_g=(double)(6*N0*N1)*(double)(N2*N3);
      nplaq2_g=nplaq1_g;
   }
   else
   {
      nplaq1_g=(double)((6*N0+3)*N1)*(double)(N2*N3);
      nplaq2_g=(double)(6*N0*N1)*(double)(N2*N3);
   }

   d1_g=0.0;
   d2_g=0.0;

   if (bc_g==1)
   {
      d1_g=cos(phi_g[0]/(double)(N1))+cos(phi_g[1]/(double)(N1))+
           cos((phi_g[0]+phi_g[1])/(double)(N1))+
           cos(phi_g[0]/(double)(N2))+cos(phi_g[1]/(double)(N2))+
           cos((phi_g[0]+phi_g[1])/(double)(N2))+
           cos(phi_g[0]/(double)(N3))+cos(phi_g[1]/(double)(N3))+
           cos((phi_g[0]+phi_g[1])/(double)(N3));
      d1_g=(d1_g-9.0)*(double)(N1*N2*N3);
   }

   if ((bc_g==1)||(bc_g==2))
   {
      d2_g=cos(phi_prime_g[0]/(double)(N1))+cos(phi_prime_g[1]/(double)(N1))+
           cos((phi_prime_g[0]+phi_prime_g[1])/(double)(N1))+
           cos(phi_prime_g[0]/(double)(N2))+cos(phi_prime_g[1]/(double)(N2))+
           cos((phi_prime_g[0]+phi_prime_g[1])/(double)(N2))+
           cos(phi_prime_g[0]/(double)(N3))+cos(phi_prime_g[1]/(double)(N3))+
           cos((phi_prime_g[0]+phi_prime_g[1])/(double)(N3));
      d2_g=(d2_g-9.0)*(double)(N1*N2*N3);
   }
}


TEST(PlaqSum, UnitField)
{
   double p1,p2,exp1,exp2;

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   exp1=3.0*nplaq1_g+d1_g+d2_g;
   exp2=3.0*nplaq2_g+d1_g+d2_g;

   EXPECT_NEAR(p1, exp1, 1.0e-15*fabs(exp1)+1.0e-14);
   EXPECT_NEAR(p2, exp2, 1.0e-15*fabs(exp2)+1.0e-14);
}


TEST(PlaqSum, ActionSlices)
{
   int t;
   double p2,act1,sum,d;
   double asl[N0];
   static su3_dble *udb;
   udb=udfld();

   random_ud();
   #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

   p2=plaq_wsum_dble(1);
   act1=plaq_action_slices(asl);

   /* act1 must equal 2*(3*nplaq2 - p2) */
   EXPECT_NEAR(act1, 2.0*(3.0*nplaq2_g-p2), 1.0e-15*fabs(act1)+1.0e-14);

   /* for bc==0 or bc==3 act1 must equal the sum of slices */
   if ((bc_g==0)||(bc_g==3))
   {
      sum=0.0;
      for (t=0;t<N0;t++) sum+=asl[t];
      d=fabs(act1-sum);
      EXPECT_NEAR(d, 0.0, 1.0e-10*fabs(act1)+1.0e-14);
   }
}


TEST(PlaqSum, GaugeInvariance)
{
   int t;
   double p1,p2,d1,d2,d3;
   double asl1[N0],asl2[N0];
   static su3_dble *udb;
   udb=udfld();

   random_ud();
   #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   plaq_action_slices(asl1);

   random_gtrans();
   apply_gtrans2ud();
   #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

   d1=fabs(p1-plaq_sum_dble(1));
   d2=fabs(p2-plaq_wsum_dble(1));
   plaq_action_slices(asl2);
   d3=0.0;
   for (t=0;t<N0;t++) d3+=fabs(asl1[t]-asl2[t]);
   d3 = d3/(double)(N0);

   EXPECT_NEAR(d1, 0.0, 1.0e-14*fabs(p1));
   EXPECT_NEAR(d2, 0.0, 1.0e-14*fabs(p2));
   EXPECT_NEAR(d3, 0.0, 1.0e-14*fabs(d3));
}


TEST(PlaqSum, TranslationInvariance)
{
   int n,t,s[4];
   double p1,p2,d1,d2,d3;
   double asl1[N0],asl2[N0];
   static su3_dble *udb;
   udb=udfld();

   random_ud();
   #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   plaq_action_slices(asl1);

   for (n=0;n<8;n++)
   {
      random_shift(s);
      shift_ud(s);
      #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

      d1=fabs(p1-plaq_sum_dble(1));
      d2=fabs(p2-plaq_wsum_dble(1));
      plaq_action_slices(asl2);
      d3=0.0;
      for (t=0;t<N0;t++)
         d3+=fabs(asl1[safe_mod(t-s[0],N0)]-asl2[t]);
      d3 = d3/(double)(N0);

      for (t=0;t<N0;t++)
         asl1[t]=asl2[t];

      EXPECT_NEAR(d1, 0.0, 1.0e-14*fabs(p1));
      EXPECT_NEAR(d2, 0.0, 1.0e-14*fabs(p2));
      EXPECT_NEAR(d3, 0.0, 1.0e-14*fabs(d3));
   }
}


TEST(PlaqSum, SumVsWsum)
{
   double p1,p2,expected;
   static su3_dble *udb;
   udb=udfld();

   if (bc_g!=1) return;

   random_ud();
   #pragma omp target update to(udb[:4*VOLUME+7*(BNDRY/4)])

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   expected=p1-9.0*(double)(N1*N2*N3);

   EXPECT_NEAR(p2, expected, 1.0e-14);
}


static mt_test_t tests[] = {
   MT_TEST(PlaqSum, UnitField),
   MT_TEST(PlaqSum, ActionSlices),
   MT_TEST(PlaqSum, GaugeInvariance),
   MT_TEST(PlaqSum, TranslationInvariance),
   MT_TEST(PlaqSum, SumVsWsum),
};


int main(int argc,char *argv[])
{
   int my_rank,bc;
   double theta[3];

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");
      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check4.c]",
                    "Syntax: check4 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   bc_g=bc;

   phi_g[0]=0.123;
   phi_g[1]=-0.534;
   phi_prime_g[0]=0.912;
   phi_prime_g[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi_g,phi_prime_g,theta);
   print_bc_parms(0x0);

   start_ranlux(0,12345);
   geometry();
   compute_expected();

   int result=RUN_ALL_TESTS(my_rank,tests);

   MPI_Finalize();
   return result;
}
