/*******************************************************************************
* 
* Atomic su3_alg_dble operations for use in OpenMP target regions.
* These are GPU-specific function-form counterparts to the macros in su3.h
* (_su3_alg_add/sub/mul_assign) and are only used by gpu/forces/force0.c.
* 
******************************************************************************/

#ifndef SU3_ALG_OPS_H
#define SU3_ALG_OPS_H

#include "su3.h"


/* r+=s */
static inline void su3_alg_add_assign(su3_alg_dble *r, su3_alg_dble s)
{
   #pragma omp atomic
   (*r).c1 += s.c1;
   #pragma omp atomic
   (*r).c2 += s.c2;
   #pragma omp atomic
   (*r).c3 += s.c3;
   #pragma omp atomic
   (*r).c4 += s.c4;
   #pragma omp atomic
   (*r).c5 += s.c5;
   #pragma omp atomic
   (*r).c6 += s.c6;
   #pragma omp atomic
   (*r).c7 += s.c7;
   #pragma omp atomic
   (*r).c8 += s.c8;
}

/* r-=s */
static inline void su3_alg_sub_assign(su3_alg_dble *r, su3_alg_dble s)
{
   #pragma omp atomic
   (*r).c1 -= s.c1;
   #pragma omp atomic
   (*r).c2 -= s.c2;
   #pragma omp atomic
   (*r).c3 -= s.c3;
   #pragma omp atomic
   (*r).c4 -= s.c4;
   #pragma omp atomic
   (*r).c5 -= s.c5;
   #pragma omp atomic
   (*r).c6 -= s.c6;
   #pragma omp atomic
   (*r).c7 -= s.c7;
   #pragma omp atomic
   (*r).c8 -= s.c8;
}

/* r+=c*s, c real */
static inline void su3_alg_mul_add_assign(su3_alg_dble *r, double c, su3_alg_dble s)
{
   #pragma omp atomic
   (*r).c1 += c * s.c1;
   #pragma omp atomic
   (*r).c2 += c * s.c2;
   #pragma omp atomic
   (*r).c3 += c * s.c3;
   #pragma omp atomic
   (*r).c4 += c * s.c4;
   #pragma omp atomic
   (*r).c5 += c * s.c5;
   #pragma omp atomic
   (*r).c6 += c * s.c6;
   #pragma omp atomic
   (*r).c7 += c * s.c7;
   #pragma omp atomic
   (*r).c8 += c * s.c8;
}

/* r-=c*s, c real */
static inline void su3_alg_mul_sub_assign(su3_alg_dble *r, double c, su3_alg_dble s)
{
   #pragma omp atomic
   (*r).c1 -= c * s.c1;
   #pragma omp atomic
   (*r).c2 -= c * s.c2;
   #pragma omp atomic
   (*r).c3 -= c * s.c3;
   #pragma omp atomic
   (*r).c4 -= c * s.c4;
   #pragma omp atomic
   (*r).c5 -= c * s.c5;
   #pragma omp atomic
   (*r).c6 -= c * s.c6;
   #pragma omp atomic
   (*r).c7 -= c * s.c7;
   #pragma omp atomic
   (*r).c8 -= c * s.c8;
}

#endif /* SU3_ALG_OPS_H */
