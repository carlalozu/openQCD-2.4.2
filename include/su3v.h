
/*******************************************************************************
*
* File su3v.h
*
* Copyright (C) 2026 Carla Lopez
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Type definitions and macros for vectorized SU(3) matrices, SU(3) vectors
*
*******************************************************************************/

#ifndef SU3V_H
#define SU3V_H

#include "global.h"
#include "lattice.h"


typedef struct
{
    size_t volume;
    double *base;
} doublev;

typedef struct
{
    size_t volume;
    double *base;
    double *re, *im;
} complexv;

typedef struct
{
    size_t volume;
    double *base;
    double *c1re, *c1im;
    double *c2re, *c2im;
    double *c3re, *c3im;
} su3_vec_field;

typedef struct
{
    su3_vec_field c1, c2, c3;

} su3_mat_field;

void su3_vec_field_init(su3_vec_field *v, size_t volume)
{
    // Round up to nearest 8
    size_t padded_volume = (volume + 7) & ~7; 
    v->volume = padded_volume;
    size_t size = 6 * padded_volume * sizeof(double);
    v->base = (double*)aligned_alloc(ALIGN, size);
    if (!v->base) {
        v->volume = 0;
        fprintf(stderr, "Erorr allocating su3_vec_field");
        abort();
    }

    v->c1re = v->base + 0*padded_volume;
    v->c1im = v->base + 1*padded_volume;
    v->c2re = v->base + 2*padded_volume;
    v->c2im = v->base + 3*padded_volume;
    v->c3re = v->base + 4*padded_volume;
    v->c3im = v->base + 5*padded_volume;
}

void su3_mat_field_init(su3_mat_field *m, size_t volume)
{
    su3_vec_field_init(&m->c1, volume);
    su3_vec_field_init(&m->c2, volume);
    su3_vec_field_init(&m->c3, volume);
}


#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matxsu3mat_retrace(
   const su3_mat_field *u, const su3_mat_field *v, doublev *res, const size_t i)
{
    double tr_1 = 0.0;
    double tr_2 = 0.0;
    double tr_3 = 0.0;

    tr_1 += u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i];
    tr_1 += u->c1.c2re[i] * v->c2.c1re[i] - u->c1.c2im[i] * v->c2.c1im[i];
    tr_1 += u->c1.c3re[i] * v->c3.c1re[i] - u->c1.c3im[i] * v->c3.c1im[i];

    tr_2 += u->c2.c1re[i] * v->c1.c2re[i] - u->c2.c1im[i] * v->c1.c2im[i];
    tr_2 += u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i];
    tr_2 += u->c2.c3re[i] * v->c3.c2re[i] - u->c2.c3im[i] * v->c3.c2im[i];

    tr_3 += u->c3.c1re[i] * v->c1.c3re[i] - u->c3.c1im[i] * v->c1.c3im[i];
    tr_3 += u->c3.c2re[i] * v->c2.c3re[i] - u->c3.c2im[i] * v->c2.c3im[i];
    tr_3 += u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];

   res->base[i] = tr_1 + tr_2 + tr_3;
}
#pragma omp end declare target

#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matxsu3mat(
   const su3_mat_field *restrict u, const su3_mat_field *restrict v, su3_mat_field *restrict res, const int n, const size_t i)
{
    int ip0 = plaq_uidx0(n,i);
    int ip1 = plaq_uidx1(n,i);

   res->c1.c1re[i] = u->c1.c1re[ip0] * v->c1.c1re[ip1] - u->c1.c1im[ip0] * v->c1.c1im[ip1] +
                      u->c1.c2re[ip0] * v->c1.c2re[ip1] - u->c1.c2im[ip0] * v->c1.c2im[ip1] +
                      u->c1.c3re[ip0] * v->c1.c3re[ip1] - u->c1.c3im[ip0] * v->c1.c3im[ip1];
   res->c1.c1im[i] = u->c1.c1re[ip0] * v->c1.c1im[ip1] + u->c1.c1im[ip0] * v->c1.c1re[ip1] +
                      u->c1.c2re[ip0] * v->c1.c2im[ip1] + u->c1.c2im[ip0] * v->c1.c2re[ip1] +
                      u->c1.c3re[ip0] * v->c1.c3im[ip1] + u->c1.c3im[ip0] * v->c1.c3re[ip1];
   res->c1.c2re[i] = u->c2.c1re[ip0] * v->c1.c1re[ip1] - u->c2.c1im[ip0] * v->c1.c1im[ip1] +
                      u->c2.c2re[ip0] * v->c1.c2re[ip1] - u->c2.c2im[ip0] * v->c1.c2im[ip1] +
                      u->c2.c3re[ip0] * v->c1.c3re[ip1] - u->c2.c3im[ip0] * v->c1.c3im[ip1];
   res->c1.c2im[i] = u->c2.c1re[ip0] * v->c1.c1im[ip1] + u->c2.c1im[ip0] * v->c1.c1re[ip1] +
                      u->c2.c2re[ip0] * v->c1.c2im[ip1] + u->c2.c2im[ip0] * v->c1.c2re[ip1] +
                      u->c2.c3re[ip0] * v->c1.c3im[ip1] + u->c2.c3im[ip0] * v->c1.c3re[ip1];
   res->c1.c3re[i] = u->c3.c1re[ip0] * v->c1.c1re[ip1] - u->c3.c1im[ip0] * v->c1.c1im[ip1] +
                      u->c3.c2re[ip0] * v->c1.c2re[ip1] - u->c3.c2im[ip0] * v->c1.c2im[ip1] +
                      u->c3.c3re[ip0] * v->c1.c3re[ip1] - u->c3.c3im[ip0] * v->c1.c3im[ip1];
   res->c1.c3im[i] = u->c3.c1re[ip0] * v->c1.c1im[ip1] + u->c3.c1im[ip0] * v->c1.c1re[ip1] +
                      u->c3.c2re[ip0] * v->c1.c2im[ip1] + u->c3.c2im[ip0] * v->c1.c2re[ip1] +
                      u->c3.c3re[ip0] * v->c1.c3im[ip1] + u->c3.c3im[ip0] * v->c1.c3re[ip1];

   res->c2.c1re[i] = u->c1.c1re[ip0] * v->c2.c1re[ip1] - u->c1.c1im[ip0] * v->c2.c1im[ip1] +
                      u->c1.c2re[ip0] * v->c2.c2re[ip1] - u->c1.c2im[ip0] * v->c2.c2im[ip1] +
                      u->c1.c3re[ip0] * v->c2.c3re[ip1] - u->c1.c3im[ip0] * v->c2.c3im[ip1];
   res->c2.c1im[i] = u->c1.c1re[ip0] * v->c2.c1im[ip1] + u->c1.c1im[ip0] * v->c2.c1re[ip1] +
                      u->c1.c2re[ip0] * v->c2.c2im[ip1] + u->c1.c2im[ip0] * v->c2.c2re[ip1] +
                      u->c1.c3re[ip0] * v->c2.c3im[ip1] + u->c1.c3im[ip0] * v->c2.c3re[ip1];
   res->c2.c2re[i] = u->c2.c1re[ip0] * v->c2.c1re[ip1] - u->c2.c1im[ip0] * v->c2.c1im[ip1] +
                      u->c2.c2re[ip0] * v->c2.c2re[ip1] - u->c2.c2im[ip0] * v->c2.c2im[ip1] +
                      u->c2.c3re[ip0] * v->c2.c3re[ip1] - u->c2.c3im[ip0] * v->c2.c3im[ip1];
   res->c2.c2im[i] = u->c2.c1re[ip0] * v->c2.c1im[ip1] + u->c2.c1im[ip0] * v->c2.c1re[ip1] +
                      u->c2.c2re[ip0] * v->c2.c2im[ip1] + u->c2.c2im[ip0] * v->c2.c2re[ip1] +
                      u->c2.c3re[ip0] * v->c2.c3im[ip1] + u->c2.c3im[ip0] * v->c2.c3re[ip1];
   res->c2.c3re[i] = u->c3.c1re[ip0] * v->c2.c1re[ip1] - u->c3.c1im[ip0] * v->c2.c1im[ip1] +
                      u->c3.c2re[ip0] * v->c2.c2re[ip1] - u->c3.c2im[ip0] * v->c2.c2im[ip1] +
                      u->c3.c3re[ip0] * v->c2.c3re[ip1] - u->c3.c3im[ip0] * v->c2.c3im[ip1];
   res->c2.c3im[i] = u->c3.c1re[ip0] * v->c2.c1im[ip1] + u->c3.c1im[ip0] * v->c2.c1re[ip1] +
                      u->c3.c2re[ip0] * v->c2.c2im[ip1] + u->c3.c2im[ip0] * v->c2.c2re[ip1] +
                      u->c3.c3re[ip0] * v->c2.c3im[ip1] + u->c3.c3im[ip0] * v->c2.c3re[ip1];

   res->c3.c1re[i] = u->c1.c1re[ip0] * v->c3.c1re[ip1] - u->c1.c1im[ip0] * v->c3.c1im[ip1] +
                      u->c1.c2re[ip0] * v->c3.c2re[ip1] - u->c1.c2im[ip0] * v->c3.c2im[ip1] +
                      u->c1.c3re[ip0] * v->c3.c3re[ip1] - u->c1.c3im[ip0] * v->c3.c3im[ip1];
   res->c3.c1im[i] = u->c1.c1re[ip0] * v->c3.c1im[ip1] + u->c1.c1im[ip0] * v->c3.c1re[ip1] +
                      u->c1.c2re[ip0] * v->c3.c2im[ip1] + u->c1.c2im[ip0] * v->c3.c2re[ip1] +
                      u->c1.c3re[ip0] * v->c3.c3im[ip1] + u->c1.c3im[ip0] * v->c3.c3re[ip1];
   res->c3.c2re[i] = u->c2.c1re[ip0] * v->c3.c1re[ip1] - u->c2.c1im[ip0] * v->c3.c1im[ip1] +
                      u->c2.c2re[ip0] * v->c3.c2re[ip1] - u->c2.c2im[ip0] * v->c3.c2im[ip1] +
                      u->c2.c3re[ip0] * v->c3.c3re[ip1] - u->c2.c3im[ip0] * v->c3.c3im[ip1];
   res->c3.c2im[i] = u->c2.c1re[ip0] * v->c3.c1im[ip1] + u->c2.c1im[ip0] * v->c3.c1re[ip1] +
                      u->c2.c2re[ip0] * v->c3.c2im[ip1] + u->c2.c2im[ip0] * v->c3.c2re[ip1] +
                      u->c2.c3re[ip0] * v->c3.c3im[ip1] + u->c2.c3im[ip0] * v->c3.c3re[ip1];
   res->c3.c3re[i] = u->c3.c1re[ip0] * v->c3.c1re[ip1] - u->c3.c1im[ip0] * v->c3.c1im[ip1] +
                      u->c3.c2re[ip0] * v->c3.c2re[ip1] - u->c3.c2im[ip0] * v->c3.c2im[ip1] +
                      u->c3.c3re[ip0] * v->c3.c3re[ip1] - u->c3.c3im[ip0] * v->c3.c3im[ip1];
   res->c3.c3im[i] = u->c3.c1re[ip0] * v->c3.c1im[ip1] + u->c3.c1im[ip0] * v->c3.c1re[ip1] +
                      u->c3.c2re[ip0] * v->c3.c2im[ip1] + u->c3.c2im[ip0] * v->c3.c2re[ip1] +
                      u->c3.c3re[ip0] * v->c3.c3im[ip1] + u->c3.c3im[ip0] * v->c3.c3re[ip1];
}
#pragma omp end declare target

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matdagxsu3matdag(
   const su3_mat_field *restrict u, const su3_mat_field *restrict v, su3_mat_field *restrict res, const int n, const size_t i)
{
   int ip2 = plaq_uidx2(n,i);
   int ip3 = plaq_uidx3(n,i);

   res->c1.c1re[i] = u->c1.c1re[ip2] * v->c1.c1re[ip3] + u->c1.c1im[ip2] * -v->c1.c1im[ip3] +
                     u->c2.c1re[ip2] * v->c1.c2re[ip3] + u->c2.c1im[ip2] * -v->c1.c2im[ip3] +
                     u->c3.c1re[ip2] * v->c1.c3re[ip3] + u->c3.c1im[ip2] * -v->c1.c3im[ip3];
   res->c1.c1im[i] = u->c1.c1re[ip2] * -v->c1.c1im[ip3] - u->c1.c1im[ip2] * v->c1.c1re[ip3] +
                     u->c2.c1re[ip2] * -v->c1.c2im[ip3] - u->c2.c1im[ip2] * v->c1.c2re[ip3] +
                     u->c3.c1re[ip2] * -v->c1.c3im[ip3] - u->c3.c1im[ip2] * v->c1.c3re[ip3];
   res->c2.c1re[i] = u->c1.c2re[ip2] * v->c1.c1re[ip3] + u->c1.c2im[ip2] * -v->c1.c1im[ip3] +
                     u->c2.c2re[ip2] * v->c1.c2re[ip3] + u->c2.c2im[ip2] * -v->c1.c2im[ip3] +
                     u->c3.c2re[ip2] * v->c1.c3re[ip3] + u->c3.c2im[ip2] * -v->c1.c3im[ip3];
   res->c2.c1im[i] = u->c1.c2re[ip2] * -v->c1.c1im[ip3] - u->c1.c2im[ip2] * v->c1.c1re[ip3] +
                     u->c2.c2re[ip2] * -v->c1.c2im[ip3] - u->c2.c2im[ip2] * v->c1.c2re[ip3] +
                     u->c3.c2re[ip2] * -v->c1.c3im[ip3] - u->c3.c2im[ip2] * v->c1.c3re[ip3];
   res->c3.c1re[i] = u->c1.c3re[ip2] * v->c1.c1re[ip3] + u->c1.c3im[ip2] * -v->c1.c1im[ip3] +
                     u->c2.c3re[ip2] * v->c1.c2re[ip3] + u->c2.c3im[ip2] * -v->c1.c2im[ip3] +
                     u->c3.c3re[ip2] * v->c1.c3re[ip3] + u->c3.c3im[ip2] * -v->c1.c3im[ip3];
   res->c3.c1im[i] = u->c1.c3re[ip2] * -v->c1.c1im[ip3] - u->c1.c3im[ip2] * v->c1.c1re[ip3] +
                     u->c2.c3re[ip2] * -v->c1.c2im[ip3] - u->c2.c3im[ip2] * v->c1.c2re[ip3] +
                     u->c3.c3re[ip2] * -v->c1.c3im[ip3] - u->c3.c3im[ip2] * v->c1.c3re[ip3];

   res->c1.c2re[i] = u->c1.c1re[ip2] * v->c2.c1re[ip3] + u->c1.c1im[ip2] * -v->c2.c1im[ip3] +
                     u->c2.c1re[ip2] * v->c2.c2re[ip3] + u->c2.c1im[ip2] * -v->c2.c2im[ip3] +
                     u->c3.c1re[ip2] * v->c2.c3re[ip3] + u->c3.c1im[ip2] * -v->c2.c3im[ip3];
   res->c1.c2im[i] = u->c1.c1re[ip2] * -v->c2.c1im[ip3] - u->c1.c1im[ip2] * v->c2.c1re[ip3] +
                     u->c2.c1re[ip2] * -v->c2.c2im[ip3] - u->c2.c1im[ip2] * v->c2.c2re[ip3] +
                     u->c3.c1re[ip2] * -v->c2.c3im[ip3] - u->c3.c1im[ip2] * v->c2.c3re[ip3];
   res->c2.c2re[i] = u->c1.c2re[ip2] * v->c2.c1re[ip3] + u->c1.c2im[ip2] * -v->c2.c1im[ip3] +
                     u->c2.c2re[ip2] * v->c2.c2re[ip3] + u->c2.c2im[ip2] * -v->c2.c2im[ip3] +
                     u->c3.c2re[ip2] * v->c2.c3re[ip3] + u->c3.c2im[ip2] * -v->c2.c3im[ip3];
   res->c2.c2im[i] = u->c1.c2re[ip2] * -v->c2.c1im[ip3] - u->c1.c2im[ip2] * v->c2.c1re[ip3] +
                     u->c2.c2re[ip2] * -v->c2.c2im[ip3] - u->c2.c2im[ip2] * v->c2.c2re[ip3] +
                     u->c3.c2re[ip2] * -v->c2.c3im[ip3] - u->c3.c2im[ip2] * v->c2.c3re[ip3];
   res->c3.c2re[i] = u->c1.c3re[ip2] * v->c2.c1re[ip3] + u->c1.c3im[ip2] * -v->c2.c1im[ip3] +
                     u->c2.c3re[ip2] * v->c2.c2re[ip3] + u->c2.c3im[ip2] * -v->c2.c2im[ip3] +
                     u->c3.c3re[ip2] * v->c2.c3re[ip3] + u->c3.c3im[ip2] * -v->c2.c3im[ip3];
   res->c3.c2im[i] = u->c1.c3re[ip2] * -v->c2.c1im[ip3] - u->c1.c3im[ip2] * v->c2.c1re[ip3] +
                     u->c2.c3re[ip2] * -v->c2.c2im[ip3] - u->c2.c3im[ip2] * v->c2.c2re[ip3] +
                     u->c3.c3re[ip2] * -v->c2.c3im[ip3] - u->c3.c3im[ip2] * v->c2.c3re[ip3];

   res->c1.c3re[i] = u->c1.c1re[ip2] * v->c3.c1re[ip3] + u->c1.c1im[ip2] * -v->c3.c1im[ip3] +
                     u->c2.c1re[ip2] * v->c3.c2re[ip3] + u->c2.c1im[ip2] * -v->c3.c2im[ip3] +
                     u->c3.c1re[ip2] * v->c3.c3re[ip3] + u->c3.c1im[ip2] * -v->c3.c3im[ip3];
   res->c1.c3im[i] = u->c1.c1re[ip2] * -v->c3.c1im[ip3] - u->c1.c1im[ip2] * v->c3.c1re[ip3] +
                     u->c2.c1re[ip2] * -v->c3.c2im[ip3] - u->c2.c1im[ip2] * v->c3.c2re[ip3] +
                     u->c3.c1re[ip2] * -v->c3.c3im[ip3] - u->c3.c1im[ip2] * v->c3.c3re[ip3];
   res->c2.c3re[i] = u->c1.c2re[ip2] * v->c3.c1re[ip3] + u->c1.c2im[ip2] * -v->c3.c1im[ip3] +
                     u->c2.c2re[ip2] * v->c3.c2re[ip3] + u->c2.c2im[ip2] * -v->c3.c2im[ip3] +
                     u->c3.c2re[ip2] * v->c3.c3re[ip3] + u->c3.c2im[ip2] * -v->c3.c3im[ip3];
   res->c2.c3im[i] = u->c1.c2re[ip2] * -v->c3.c1im[ip3] - u->c1.c2im[ip2] * v->c3.c1re[ip3] +
                     u->c2.c2re[ip2] * -v->c3.c2im[ip3] - u->c2.c2im[ip2] * v->c3.c2re[ip3] +
                     u->c3.c2re[ip2] * -v->c3.c3im[ip3] - u->c3.c2im[ip2] * v->c3.c3re[ip3];
   res->c3.c3re[i] = u->c1.c3re[ip2] * v->c3.c1re[ip3] + u->c1.c3im[ip2] * -v->c3.c1im[ip3] +
                     u->c2.c3re[ip2] * v->c3.c2re[ip3] + u->c2.c3im[ip2] * -v->c3.c2im[ip3] +
                     u->c3.c3re[ip2] * v->c3.c3re[ip3] + u->c3.c3im[ip2] * -v->c3.c3im[ip3];
   res->c3.c3im[i] = u->c1.c3re[ip2] * -v->c3.c1im[ip3] - u->c1.c3im[ip2] * v->c3.c1re[ip3] +
                     u->c2.c3re[ip2] * -v->c3.c2im[ip3] - u->c2.c3im[ip2] * v->c3.c2re[ip3] +
                     u->c3.c3re[ip2] * -v->c3.c3im[ip3] - u->c3.c3im[ip2] * v->c3.c3re[ip3];
}
#pragma omp end declare target

#endif
