
/*******************************************************************************
 *
 * File su3prod.c
 *
 * Copyright (C) 2005, 2009-2013, 2016, 2018, 2021 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Products of double-precision 3x3 matrices in vectorized form
 *
 *   void fsu3matxsu3mat(
 *     const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, int n,
 *     int i)
 *   Computes w=u*v assuming that w is different from u.
 *
 *   void fsu3matdagxsu3matdag(
 *     const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, int n,
 *     int i)
 *   Computes w=u^dag*v^dag assuming that w is different from u and v.
 *
 *   void fsu3matxsu3mat_retrace(
 *     const su3_mat_field *u, const su3_mat_field *v, doublev *res, int i)
 *
 *******************************************************************************/

#ifndef SU3PRODV_C
#define SU3PRODV_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "su3v.h"
#include "su3fcts.h"

#pragma omp declare target
double fsu3matxsu3mat_retrace(
    const su3_mat_field *u, const su3_mat_field *v, int i)
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

   return tr_1 + tr_2 + tr_3;
}
#pragma omp end declare target

#pragma omp declare target
void fsu3matxsu3mat(
    const su3_mat_field *u, su3_dble *res, int ip0, int ip1)
{
   //  printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip0, ip1);

   res->c11.re = u->c1.c1re[ip0] * u->c1.c1re[ip1] - u->c1.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c1.c2re[ip1] - u->c1.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c1.c3re[ip1] - u->c1.c3im[ip0] * u->c1.c3im[ip1];
   res->c11.im = u->c1.c1re[ip0] * u->c1.c1im[ip1] + u->c1.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c1.c2im[ip1] + u->c1.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c1.c3im[ip1] + u->c1.c3im[ip0] * u->c1.c3re[ip1];
   res->c12.re = u->c2.c1re[ip0] * u->c1.c1re[ip1] - u->c2.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c1.c2re[ip1] - u->c2.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c1.c3re[ip1] - u->c2.c3im[ip0] * u->c1.c3im[ip1];
   res->c12.im = u->c2.c1re[ip0] * u->c1.c1im[ip1] + u->c2.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c1.c2im[ip1] + u->c2.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c1.c3im[ip1] + u->c2.c3im[ip0] * u->c1.c3re[ip1];
   res->c13.re = u->c3.c1re[ip0] * u->c1.c1re[ip1] - u->c3.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c1.c2re[ip1] - u->c3.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c1.c3re[ip1] - u->c3.c3im[ip0] * u->c1.c3im[ip1];
   res->c13.im = u->c3.c1re[ip0] * u->c1.c1im[ip1] + u->c3.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c1.c2im[ip1] + u->c3.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c1.c3im[ip1] + u->c3.c3im[ip0] * u->c1.c3re[ip1];

   res->c21.re = u->c1.c1re[ip0] * u->c2.c1re[ip1] - u->c1.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c2.c2re[ip1] - u->c1.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c2.c3re[ip1] - u->c1.c3im[ip0] * u->c2.c3im[ip1];
   res->c21.im = u->c1.c1re[ip0] * u->c2.c1im[ip1] + u->c1.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c2.c2im[ip1] + u->c1.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c2.c3im[ip1] + u->c1.c3im[ip0] * u->c2.c3re[ip1];
   res->c22.re = u->c2.c1re[ip0] * u->c2.c1re[ip1] - u->c2.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c2.c2re[ip1] - u->c2.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c2.c3re[ip1] - u->c2.c3im[ip0] * u->c2.c3im[ip1];
   res->c22.im = u->c2.c1re[ip0] * u->c2.c1im[ip1] + u->c2.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c2.c2im[ip1] + u->c2.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c2.c3im[ip1] + u->c2.c3im[ip0] * u->c2.c3re[ip1];
   res->c23.re = u->c3.c1re[ip0] * u->c2.c1re[ip1] - u->c3.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c2.c2re[ip1] - u->c3.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c2.c3re[ip1] - u->c3.c3im[ip0] * u->c2.c3im[ip1];
   res->c23.im = u->c3.c1re[ip0] * u->c2.c1im[ip1] + u->c3.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c2.c2im[ip1] + u->c3.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c2.c3im[ip1] + u->c3.c3im[ip0] * u->c2.c3re[ip1];

   res->c31.re = u->c1.c1re[ip0] * u->c3.c1re[ip1] - u->c1.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c3.c2re[ip1] - u->c1.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c3.c3re[ip1] - u->c1.c3im[ip0] * u->c3.c3im[ip1];
   res->c31.im = u->c1.c1re[ip0] * u->c3.c1im[ip1] + u->c1.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c3.c2im[ip1] + u->c1.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c3.c3im[ip1] + u->c1.c3im[ip0] * u->c3.c3re[ip1];
   res->c32.re = u->c2.c1re[ip0] * u->c3.c1re[ip1] - u->c2.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c3.c2re[ip1] - u->c2.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c3.c3re[ip1] - u->c2.c3im[ip0] * u->c3.c3im[ip1];
   res->c32.im = u->c2.c1re[ip0] * u->c3.c1im[ip1] + u->c2.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c3.c2im[ip1] + u->c2.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c3.c3im[ip1] + u->c2.c3im[ip0] * u->c3.c3re[ip1];
   res->c33.re = u->c3.c1re[ip0] * u->c3.c1re[ip1] - u->c3.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c3.c2re[ip1] - u->c3.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c3.c3re[ip1] - u->c3.c3im[ip0] * u->c3.c3im[ip1];
   res->c33.im = u->c3.c1re[ip0] * u->c3.c1im[ip1] + u->c3.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c3.c2im[ip1] + u->c3.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c3.c3im[ip1] + u->c3.c3im[ip0] * u->c3.c3re[ip1];
}
#pragma omp end declare target

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
#pragma omp declare target
void fsu3matdagxsu3matdag(
    const su3_mat_field *u, su3_dble *res, int ip2, int ip3)
{
   // printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip2, ip3);

   res->c11.re = u->c1.c1re[ip2] * u->c1.c1re[ip3] + u->c1.c1im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c1.c2re[ip3] + u->c2.c1im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c1.c3re[ip3] + u->c3.c1im[ip2] * -u->c1.c3im[ip3];
   res->c11.im = u->c1.c1re[ip2] * -u->c1.c1im[ip3] - u->c1.c1im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c1.c2im[ip3] - u->c2.c1im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c1.c3im[ip3] - u->c3.c1im[ip2] * u->c1.c3re[ip3];
   res->c21.re = u->c1.c2re[ip2] * u->c1.c1re[ip3] + u->c1.c2im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c1.c2re[ip3] + u->c2.c2im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c1.c3re[ip3] + u->c3.c2im[ip2] * -u->c1.c3im[ip3];
   res->c21.im = u->c1.c2re[ip2] * -u->c1.c1im[ip3] - u->c1.c2im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c1.c2im[ip3] - u->c2.c2im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c1.c3im[ip3] - u->c3.c2im[ip2] * u->c1.c3re[ip3];
   res->c31.re = u->c1.c3re[ip2] * u->c1.c1re[ip3] + u->c1.c3im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c1.c2re[ip3] + u->c2.c3im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c1.c3re[ip3] + u->c3.c3im[ip2] * -u->c1.c3im[ip3];
   res->c31.im = u->c1.c3re[ip2] * -u->c1.c1im[ip3] - u->c1.c3im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c1.c2im[ip3] - u->c2.c3im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c1.c3im[ip3] - u->c3.c3im[ip2] * u->c1.c3re[ip3];

   res->c12.re = u->c1.c1re[ip2] * u->c2.c1re[ip3] + u->c1.c1im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c2.c2re[ip3] + u->c2.c1im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c2.c3re[ip3] + u->c3.c1im[ip2] * -u->c2.c3im[ip3];
   res->c12.im = u->c1.c1re[ip2] * -u->c2.c1im[ip3] - u->c1.c1im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c2.c2im[ip3] - u->c2.c1im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c2.c3im[ip3] - u->c3.c1im[ip2] * u->c2.c3re[ip3];
   res->c22.re = u->c1.c2re[ip2] * u->c2.c1re[ip3] + u->c1.c2im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c2.c2re[ip3] + u->c2.c2im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c2.c3re[ip3] + u->c3.c2im[ip2] * -u->c2.c3im[ip3];
   res->c22.im = u->c1.c2re[ip2] * -u->c2.c1im[ip3] - u->c1.c2im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c2.c2im[ip3] - u->c2.c2im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c2.c3im[ip3] - u->c3.c2im[ip2] * u->c2.c3re[ip3];
   res->c32.re = u->c1.c3re[ip2] * u->c2.c1re[ip3] + u->c1.c3im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c2.c2re[ip3] + u->c2.c3im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c2.c3re[ip3] + u->c3.c3im[ip2] * -u->c2.c3im[ip3];
   res->c32.im = u->c1.c3re[ip2] * -u->c2.c1im[ip3] - u->c1.c3im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c2.c2im[ip3] - u->c2.c3im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c2.c3im[ip3] - u->c3.c3im[ip2] * u->c2.c3re[ip3];

   res->c13.re = u->c1.c1re[ip2] * u->c3.c1re[ip3] + u->c1.c1im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c3.c2re[ip3] + u->c2.c1im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c3.c3re[ip3] + u->c3.c1im[ip2] * -u->c3.c3im[ip3];
   res->c13.im = u->c1.c1re[ip2] * -u->c3.c1im[ip3] - u->c1.c1im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c3.c2im[ip3] - u->c2.c1im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c3.c3im[ip3] - u->c3.c1im[ip2] * u->c3.c3re[ip3];
   res->c23.re = u->c1.c2re[ip2] * u->c3.c1re[ip3] + u->c1.c2im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c3.c2re[ip3] + u->c2.c2im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c3.c3re[ip3] + u->c3.c2im[ip2] * -u->c3.c3im[ip3];
   res->c23.im = u->c1.c2re[ip2] * -u->c3.c1im[ip3] - u->c1.c2im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c3.c2im[ip3] - u->c2.c2im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c3.c3im[ip3] - u->c3.c2im[ip2] * u->c3.c3re[ip3];
   res->c33.re = u->c1.c3re[ip2] * u->c3.c1re[ip3] + u->c1.c3im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c3.c2re[ip3] + u->c2.c3im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c3.c3re[ip3] + u->c3.c3im[ip2] * -u->c3.c3im[ip3];
   res->c33.im = u->c1.c3re[ip2] * -u->c3.c1im[ip3] - u->c1.c3im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c3.c2im[ip3] - u->c2.c3im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c3.c3im[ip3] - u->c3.c3im[ip2] * u->c3.c3re[ip3];
}
#pragma omp end declare target

/*******************************************************************************
 *
 * Fused plaquette kernel - reduced register pressure version
 *
 * Key idea: instead of computing two full su3_dble matrices (wd1, wd2) and
 * then taking Re Tr(wd1 * wd2), we fuse everything into a single function
 * that computes the trace directly, one diagonal element at a time.
 *
 * Register savings:
 *   Original: wd1 (18 doubles) + wd2 (18 doubles) + intermediates ~ 140 regs
 *   Fused:    one row of A + one col of B + running trace ~ 40-50 regs
 *
 *******************************************************************************/

 
 /*
 * Computes Re Tr( (u[ip0]*v[ip1]) * (u[ip2]^dag * v[ip3]^dag) ) directly
 * without storing any intermediate su3_dble matrix.
 *
 * Re Tr(A*B) = sum_{i=1}^{3} Re( sum_{k=1}^{3} A_ik * B_ki )
 *
 * We compute one diagonal contribution at a time, keeping only
 * one row of A and one column of B live simultaneously.
 */
 #pragma omp declare target
double plaq_retrace_fused(const su3_mat_field *u,
                                  int ip0, int ip1,
                                  int ip2, int ip3)
{
   double tr = 0.0;

   double xc11re = u->c1.c1re[ip0] * u->c1.c1re[ip1] - u->c1.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c1.c2re[ip1] - u->c1.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c1.c3re[ip1] - u->c1.c3im[ip0] * u->c1.c3im[ip1];
   double xc11im = u->c1.c1re[ip0] * u->c1.c1im[ip1] + u->c1.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c1.c2im[ip1] + u->c1.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c1.c3im[ip1] + u->c1.c3im[ip0] * u->c1.c3re[ip1];
   double xc12re = u->c2.c1re[ip0] * u->c1.c1re[ip1] - u->c2.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c1.c2re[ip1] - u->c2.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c1.c3re[ip1] - u->c2.c3im[ip0] * u->c1.c3im[ip1];
   double xc12im = u->c2.c1re[ip0] * u->c1.c1im[ip1] + u->c2.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c1.c2im[ip1] + u->c2.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c1.c3im[ip1] + u->c2.c3im[ip0] * u->c1.c3re[ip1];
   double xc13re = u->c3.c1re[ip0] * u->c1.c1re[ip1] - u->c3.c1im[ip0] * u->c1.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c1.c2re[ip1] - u->c3.c2im[ip0] * u->c1.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c1.c3re[ip1] - u->c3.c3im[ip0] * u->c1.c3im[ip1];
   double xc13im = u->c3.c1re[ip0] * u->c1.c1im[ip1] + u->c3.c1im[ip0] * u->c1.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c1.c2im[ip1] + u->c3.c2im[ip0] * u->c1.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c1.c3im[ip1] + u->c3.c3im[ip0] * u->c1.c3re[ip1];

   double xc21re = u->c1.c1re[ip0] * u->c2.c1re[ip1] - u->c1.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c2.c2re[ip1] - u->c1.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c2.c3re[ip1] - u->c1.c3im[ip0] * u->c2.c3im[ip1];
   double xc21im = u->c1.c1re[ip0] * u->c2.c1im[ip1] + u->c1.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c2.c2im[ip1] + u->c1.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c2.c3im[ip1] + u->c1.c3im[ip0] * u->c2.c3re[ip1];
   double xc22re = u->c2.c1re[ip0] * u->c2.c1re[ip1] - u->c2.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c2.c2re[ip1] - u->c2.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c2.c3re[ip1] - u->c2.c3im[ip0] * u->c2.c3im[ip1];
   double xc22im = u->c2.c1re[ip0] * u->c2.c1im[ip1] + u->c2.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c2.c2im[ip1] + u->c2.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c2.c3im[ip1] + u->c2.c3im[ip0] * u->c2.c3re[ip1];
   double xc23re = u->c3.c1re[ip0] * u->c2.c1re[ip1] - u->c3.c1im[ip0] * u->c2.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c2.c2re[ip1] - u->c3.c2im[ip0] * u->c2.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c2.c3re[ip1] - u->c3.c3im[ip0] * u->c2.c3im[ip1];
   double xc23im = u->c3.c1re[ip0] * u->c2.c1im[ip1] + u->c3.c1im[ip0] * u->c2.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c2.c2im[ip1] + u->c3.c2im[ip0] * u->c2.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c2.c3im[ip1] + u->c3.c3im[ip0] * u->c2.c3re[ip1];

   double xc31re = u->c1.c1re[ip0] * u->c3.c1re[ip1] - u->c1.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c1.c2re[ip0] * u->c3.c2re[ip1] - u->c1.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c1.c3re[ip0] * u->c3.c3re[ip1] - u->c1.c3im[ip0] * u->c3.c3im[ip1];
   double xc31im = u->c1.c1re[ip0] * u->c3.c1im[ip1] + u->c1.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c1.c2re[ip0] * u->c3.c2im[ip1] + u->c1.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c1.c3re[ip0] * u->c3.c3im[ip1] + u->c1.c3im[ip0] * u->c3.c3re[ip1];
   double xc32re = u->c2.c1re[ip0] * u->c3.c1re[ip1] - u->c2.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c2.c2re[ip0] * u->c3.c2re[ip1] - u->c2.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c2.c3re[ip0] * u->c3.c3re[ip1] - u->c2.c3im[ip0] * u->c3.c3im[ip1];
   double xc32im = u->c2.c1re[ip0] * u->c3.c1im[ip1] + u->c2.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c2.c2re[ip0] * u->c3.c2im[ip1] + u->c2.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c2.c3re[ip0] * u->c3.c3im[ip1] + u->c2.c3im[ip0] * u->c3.c3re[ip1];
   double xc33re = u->c3.c1re[ip0] * u->c3.c1re[ip1] - u->c3.c1im[ip0] * u->c3.c1im[ip1] +
                 u->c3.c2re[ip0] * u->c3.c2re[ip1] - u->c3.c2im[ip0] * u->c3.c2im[ip1] +
                 u->c3.c3re[ip0] * u->c3.c3re[ip1] - u->c3.c3im[ip0] * u->c3.c3im[ip1];
   double xc33im = u->c3.c1re[ip0] * u->c3.c1im[ip1] + u->c3.c1im[ip0] * u->c3.c1re[ip1] +
                 u->c3.c2re[ip0] * u->c3.c2im[ip1] + u->c3.c2im[ip0] * u->c3.c2re[ip1] +
                 u->c3.c3re[ip0] * u->c3.c3im[ip1] + u->c3.c3im[ip0] * u->c3.c3re[ip1];
                 
   double yc11re = u->c1.c1re[ip2] * u->c1.c1re[ip3] + u->c1.c1im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c1.c2re[ip3] + u->c2.c1im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c1.c3re[ip3] + u->c3.c1im[ip2] * -u->c1.c3im[ip3];
   double yc11im = u->c1.c1re[ip2] * -u->c1.c1im[ip3] - u->c1.c1im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c1.c2im[ip3] - u->c2.c1im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c1.c3im[ip3] - u->c3.c1im[ip2] * u->c1.c3re[ip3];
   double yc21re = u->c1.c2re[ip2] * u->c1.c1re[ip3] + u->c1.c2im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c1.c2re[ip3] + u->c2.c2im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c1.c3re[ip3] + u->c3.c2im[ip2] * -u->c1.c3im[ip3];
   double yc21im = u->c1.c2re[ip2] * -u->c1.c1im[ip3] - u->c1.c2im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c1.c2im[ip3] - u->c2.c2im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c1.c3im[ip3] - u->c3.c2im[ip2] * u->c1.c3re[ip3];
   double yc31re = u->c1.c3re[ip2] * u->c1.c1re[ip3] + u->c1.c3im[ip2] * -u->c1.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c1.c2re[ip3] + u->c2.c3im[ip2] * -u->c1.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c1.c3re[ip3] + u->c3.c3im[ip2] * -u->c1.c3im[ip3];
   double yc31im = u->c1.c3re[ip2] * -u->c1.c1im[ip3] - u->c1.c3im[ip2] * u->c1.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c1.c2im[ip3] - u->c2.c3im[ip2] * u->c1.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c1.c3im[ip3] - u->c3.c3im[ip2] * u->c1.c3re[ip3];

    tr += xc11re * yc11re - xc11im * yc11im;
    tr += xc12re * yc21re - xc12im * yc21im;
    tr += xc13re * yc31re - xc13im * yc31im;

   double yc12re = u->c1.c1re[ip2] * u->c2.c1re[ip3] + u->c1.c1im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c2.c2re[ip3] + u->c2.c1im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c2.c3re[ip3] + u->c3.c1im[ip2] * -u->c2.c3im[ip3];
   double yc12im = u->c1.c1re[ip2] * -u->c2.c1im[ip3] - u->c1.c1im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c2.c2im[ip3] - u->c2.c1im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c2.c3im[ip3] - u->c3.c1im[ip2] * u->c2.c3re[ip3];
   double yc22re = u->c1.c2re[ip2] * u->c2.c1re[ip3] + u->c1.c2im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c2.c2re[ip3] + u->c2.c2im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c2.c3re[ip3] + u->c3.c2im[ip2] * -u->c2.c3im[ip3];
   double yc22im = u->c1.c2re[ip2] * -u->c2.c1im[ip3] - u->c1.c2im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c2.c2im[ip3] - u->c2.c2im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c2.c3im[ip3] - u->c3.c2im[ip2] * u->c2.c3re[ip3];
   double yc32re = u->c1.c3re[ip2] * u->c2.c1re[ip3] + u->c1.c3im[ip2] * -u->c2.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c2.c2re[ip3] + u->c2.c3im[ip2] * -u->c2.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c2.c3re[ip3] + u->c3.c3im[ip2] * -u->c2.c3im[ip3];
   double yc32im = u->c1.c3re[ip2] * -u->c2.c1im[ip3] - u->c1.c3im[ip2] * u->c2.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c2.c2im[ip3] - u->c2.c3im[ip2] * u->c2.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c2.c3im[ip3] - u->c3.c3im[ip2] * u->c2.c3re[ip3];

    tr += xc21re * yc12re - xc21im * yc12im;
    tr += xc22re * yc22re - xc22im * yc22im;
    tr += xc23re * yc32re - xc23im * yc32im;

   double yc13re = u->c1.c1re[ip2] * u->c3.c1re[ip3] + u->c1.c1im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c1re[ip2] * u->c3.c2re[ip3] + u->c2.c1im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c1re[ip2] * u->c3.c3re[ip3] + u->c3.c1im[ip2] * -u->c3.c3im[ip3];
   double yc13im = u->c1.c1re[ip2] * -u->c3.c1im[ip3] - u->c1.c1im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c1re[ip2] * -u->c3.c2im[ip3] - u->c2.c1im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c1re[ip2] * -u->c3.c3im[ip3] - u->c3.c1im[ip2] * u->c3.c3re[ip3];
   double yc23re = u->c1.c2re[ip2] * u->c3.c1re[ip3] + u->c1.c2im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c2re[ip2] * u->c3.c2re[ip3] + u->c2.c2im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c2re[ip2] * u->c3.c3re[ip3] + u->c3.c2im[ip2] * -u->c3.c3im[ip3];
   double yc23im = u->c1.c2re[ip2] * -u->c3.c1im[ip3] - u->c1.c2im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c2re[ip2] * -u->c3.c2im[ip3] - u->c2.c2im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c2re[ip2] * -u->c3.c3im[ip3] - u->c3.c2im[ip2] * u->c3.c3re[ip3];
   double yc33re = u->c1.c3re[ip2] * u->c3.c1re[ip3] + u->c1.c3im[ip2] * -u->c3.c1im[ip3] +
                 u->c2.c3re[ip2] * u->c3.c2re[ip3] + u->c2.c3im[ip2] * -u->c3.c2im[ip3] +
                 u->c3.c3re[ip2] * u->c3.c3re[ip3] + u->c3.c3im[ip2] * -u->c3.c3im[ip3];
   double yc33im = u->c1.c3re[ip2] * -u->c3.c1im[ip3] - u->c1.c3im[ip2] * u->c3.c1re[ip3] +
                 u->c2.c3re[ip2] * -u->c3.c2im[ip3] - u->c2.c3im[ip2] * u->c3.c2re[ip3] +
                 u->c3.c3re[ip2] * -u->c3.c3im[ip3] - u->c3.c3im[ip2] * u->c3.c3re[ip3];

    tr += xc31re * yc13re - xc31im * yc13im;
    tr += xc32re * yc23re - xc32im * yc23im;
    tr += xc33re * yc33re - xc33im * yc33im;

   return tr;
}
#pragma omp end declare target

#endif
