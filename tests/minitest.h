/*******************************************************************************
*
* minitest.h -- minimal Google-Test-style unit testing for MPI/C programs.
*
* Usage:
*   1. Include this header.
*   2. Define tests with  TEST(Suite, Case) { ... }.
*   3. Register them in a static array:
*        static mt_test_t tests[] = { MT_TEST(Suite, Case), ... };
*   4. In main(), call  return RUN_ALL_TESTS(my_rank, tests);
*
* Assertions: EXPECT_TRUE, EXPECT_EQ, EXPECT_NEAR.
* Skip a test with  SKIP_TEST("reason").
* Each test is collective: MPI_Allreduce gathers failures across all ranks.
* Output is printed only on rank 0.
*
*******************************************************************************/

#pragma once

#include <stdio.h>
#include <math.h>
#include "mpi.h"

typedef struct {
   const char *name;
   void       (*fn)(void);
} mt_test_t;

static int mt_failures_;
static int mt_skipped_;
static int mt_rank_;

#define TEST(suite, name)   static void mt_fn_##suite##_##name(void)
#define MT_TEST(suite, name) { #suite "." #name, mt_fn_##suite##_##name }

#define EXPECT_TRUE(expr) \
   do { if (!(expr)) { \
      if (mt_rank_==0) \
         printf("[   FAIL   ] %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      mt_failures_++; \
   } } while (0)

#define EXPECT_EQ(a, b) \
   do { if ((a)!=(b)) { \
      if (mt_rank_==0) \
         printf("[   FAIL   ] %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
      mt_failures_++; \
   } } while (0)

#define MT_PRINT(fmt, ...) \
   printf("[   INFO   ] " fmt "\n", ##__VA_ARGS__)

#define SKIP_TEST(msg) \
   do { mt_skipped_=1; \
      if (mt_rank_==0) \
         printf("[   INFO   ] %s\n", (msg)); \
      return; \
   } while (0)

#define EXPECT_NEAR(a, b, tol) \
   do { \
      double _a=(double)(a), _b=(double)(b), _t=(double)(tol); \
      if (fabs(_a-_b)>_t) { \
         if (mt_rank_==0) \
            printf("[   FAIL   ] %s:%d: |%.10g - %.10g| = %.2e, tol=%.2e\n", \
                   __FILE__, __LINE__, _a, _b, fabs(_a-_b), _t); \
         mt_failures_++; \
      } \
   } while (0)

static inline int mt_run_(mt_test_t *tests, int n, int rank)
{
   int i, pass=0, fail=0, skip=0, gfail, gskip;
   mt_rank_=rank;

   if (rank==0) printf("[==========] Running %d test(s).\n", n);

   for (i=0; i<n; i++) {
      if (rank==0) printf("[ RUN      ] %s\n", tests[i].name);
      mt_failures_=0;
      mt_skipped_=0;
      tests[i].fn();
      MPI_Allreduce(&mt_failures_, &gfail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&mt_skipped_,  &gskip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (gskip) {
         if (rank==0) printf("[  SKIPPED ] %s\n", tests[i].name);
         skip++;
      } else if (gfail==0) {
         if (rank==0) printf("[       OK ] %s\n", tests[i].name);
         pass++;
      } else {
         if (rank==0) printf("[  FAILED  ] %s\n", tests[i].name);
         fail++;
      }
   }

   if (rank==0) {
      printf("[==========] %d test(s) ran.\n", n);
      printf("[  PASSED  ] %d test(s).\n", pass);
      if (skip) printf("[  SKIPPED ] %d test(s).\n", skip);
      if (fail) printf("[  FAILED  ] %d test(s).\n", fail);
   }

   return fail;
}

#define RUN_ALL_TESTS(rank, tests) \
   mt_run_((tests), (int)(sizeof(tests)/sizeof((tests)[0])), (rank))
