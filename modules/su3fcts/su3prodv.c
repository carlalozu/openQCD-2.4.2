
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
    const su3_mat_field *u, const su3_mat_field *v, su3_dble *res, int n, int i)
{
   int ip0 = plaq_uidx0(n, i);
   int ip1 = plaq_uidx1(n, i);

   //  printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip0, ip1);

   res->c11.re = u->c1.c1re[ip0] * v->c1.c1re[ip1] - u->c1.c1im[ip0] * v->c1.c1im[ip1] +
                 u->c1.c2re[ip0] * v->c1.c2re[ip1] - u->c1.c2im[ip0] * v->c1.c2im[ip1] +
                 u->c1.c3re[ip0] * v->c1.c3re[ip1] - u->c1.c3im[ip0] * v->c1.c3im[ip1];
   res->c11.im = u->c1.c1re[ip0] * v->c1.c1im[ip1] + u->c1.c1im[ip0] * v->c1.c1re[ip1] +
                 u->c1.c2re[ip0] * v->c1.c2im[ip1] + u->c1.c2im[ip0] * v->c1.c2re[ip1] +
                 u->c1.c3re[ip0] * v->c1.c3im[ip1] + u->c1.c3im[ip0] * v->c1.c3re[ip1];
   res->c12.re = u->c2.c1re[ip0] * v->c1.c1re[ip1] - u->c2.c1im[ip0] * v->c1.c1im[ip1] +
                 u->c2.c2re[ip0] * v->c1.c2re[ip1] - u->c2.c2im[ip0] * v->c1.c2im[ip1] +
                 u->c2.c3re[ip0] * v->c1.c3re[ip1] - u->c2.c3im[ip0] * v->c1.c3im[ip1];
   res->c12.im = u->c2.c1re[ip0] * v->c1.c1im[ip1] + u->c2.c1im[ip0] * v->c1.c1re[ip1] +
                 u->c2.c2re[ip0] * v->c1.c2im[ip1] + u->c2.c2im[ip0] * v->c1.c2re[ip1] +
                 u->c2.c3re[ip0] * v->c1.c3im[ip1] + u->c2.c3im[ip0] * v->c1.c3re[ip1];
   res->c13.re = u->c3.c1re[ip0] * v->c1.c1re[ip1] - u->c3.c1im[ip0] * v->c1.c1im[ip1] +
                 u->c3.c2re[ip0] * v->c1.c2re[ip1] - u->c3.c2im[ip0] * v->c1.c2im[ip1] +
                 u->c3.c3re[ip0] * v->c1.c3re[ip1] - u->c3.c3im[ip0] * v->c1.c3im[ip1];
   res->c13.im = u->c3.c1re[ip0] * v->c1.c1im[ip1] + u->c3.c1im[ip0] * v->c1.c1re[ip1] +
                 u->c3.c2re[ip0] * v->c1.c2im[ip1] + u->c3.c2im[ip0] * v->c1.c2re[ip1] +
                 u->c3.c3re[ip0] * v->c1.c3im[ip1] + u->c3.c3im[ip0] * v->c1.c3re[ip1];

   res->c21.re = u->c1.c1re[ip0] * v->c2.c1re[ip1] - u->c1.c1im[ip0] * v->c2.c1im[ip1] +
                 u->c1.c2re[ip0] * v->c2.c2re[ip1] - u->c1.c2im[ip0] * v->c2.c2im[ip1] +
                 u->c1.c3re[ip0] * v->c2.c3re[ip1] - u->c1.c3im[ip0] * v->c2.c3im[ip1];
   res->c21.im = u->c1.c1re[ip0] * v->c2.c1im[ip1] + u->c1.c1im[ip0] * v->c2.c1re[ip1] +
                 u->c1.c2re[ip0] * v->c2.c2im[ip1] + u->c1.c2im[ip0] * v->c2.c2re[ip1] +
                 u->c1.c3re[ip0] * v->c2.c3im[ip1] + u->c1.c3im[ip0] * v->c2.c3re[ip1];
   res->c22.re = u->c2.c1re[ip0] * v->c2.c1re[ip1] - u->c2.c1im[ip0] * v->c2.c1im[ip1] +
                 u->c2.c2re[ip0] * v->c2.c2re[ip1] - u->c2.c2im[ip0] * v->c2.c2im[ip1] +
                 u->c2.c3re[ip0] * v->c2.c3re[ip1] - u->c2.c3im[ip0] * v->c2.c3im[ip1];
   res->c22.im = u->c2.c1re[ip0] * v->c2.c1im[ip1] + u->c2.c1im[ip0] * v->c2.c1re[ip1] +
                 u->c2.c2re[ip0] * v->c2.c2im[ip1] + u->c2.c2im[ip0] * v->c2.c2re[ip1] +
                 u->c2.c3re[ip0] * v->c2.c3im[ip1] + u->c2.c3im[ip0] * v->c2.c3re[ip1];
   res->c23.re = u->c3.c1re[ip0] * v->c2.c1re[ip1] - u->c3.c1im[ip0] * v->c2.c1im[ip1] +
                 u->c3.c2re[ip0] * v->c2.c2re[ip1] - u->c3.c2im[ip0] * v->c2.c2im[ip1] +
                 u->c3.c3re[ip0] * v->c2.c3re[ip1] - u->c3.c3im[ip0] * v->c2.c3im[ip1];
   res->c23.im = u->c3.c1re[ip0] * v->c2.c1im[ip1] + u->c3.c1im[ip0] * v->c2.c1re[ip1] +
                 u->c3.c2re[ip0] * v->c2.c2im[ip1] + u->c3.c2im[ip0] * v->c2.c2re[ip1] +
                 u->c3.c3re[ip0] * v->c2.c3im[ip1] + u->c3.c3im[ip0] * v->c2.c3re[ip1];

   res->c31.re = u->c1.c1re[ip0] * v->c3.c1re[ip1] - u->c1.c1im[ip0] * v->c3.c1im[ip1] +
                 u->c1.c2re[ip0] * v->c3.c2re[ip1] - u->c1.c2im[ip0] * v->c3.c2im[ip1] +
                 u->c1.c3re[ip0] * v->c3.c3re[ip1] - u->c1.c3im[ip0] * v->c3.c3im[ip1];
   res->c31.im = u->c1.c1re[ip0] * v->c3.c1im[ip1] + u->c1.c1im[ip0] * v->c3.c1re[ip1] +
                 u->c1.c2re[ip0] * v->c3.c2im[ip1] + u->c1.c2im[ip0] * v->c3.c2re[ip1] +
                 u->c1.c3re[ip0] * v->c3.c3im[ip1] + u->c1.c3im[ip0] * v->c3.c3re[ip1];
   res->c32.re = u->c2.c1re[ip0] * v->c3.c1re[ip1] - u->c2.c1im[ip0] * v->c3.c1im[ip1] +
                 u->c2.c2re[ip0] * v->c3.c2re[ip1] - u->c2.c2im[ip0] * v->c3.c2im[ip1] +
                 u->c2.c3re[ip0] * v->c3.c3re[ip1] - u->c2.c3im[ip0] * v->c3.c3im[ip1];
   res->c32.im = u->c2.c1re[ip0] * v->c3.c1im[ip1] + u->c2.c1im[ip0] * v->c3.c1re[ip1] +
                 u->c2.c2re[ip0] * v->c3.c2im[ip1] + u->c2.c2im[ip0] * v->c3.c2re[ip1] +
                 u->c2.c3re[ip0] * v->c3.c3im[ip1] + u->c2.c3im[ip0] * v->c3.c3re[ip1];
   res->c33.re = u->c3.c1re[ip0] * v->c3.c1re[ip1] - u->c3.c1im[ip0] * v->c3.c1im[ip1] +
                 u->c3.c2re[ip0] * v->c3.c2re[ip1] - u->c3.c2im[ip0] * v->c3.c2im[ip1] +
                 u->c3.c3re[ip0] * v->c3.c3re[ip1] - u->c3.c3im[ip0] * v->c3.c3im[ip1];
   res->c33.im = u->c3.c1re[ip0] * v->c3.c1im[ip1] + u->c3.c1im[ip0] * v->c3.c1re[ip1] +
                 u->c3.c2re[ip0] * v->c3.c2im[ip1] + u->c3.c2im[ip0] * v->c3.c2re[ip1] +
                 u->c3.c3re[ip0] * v->c3.c3im[ip1] + u->c3.c3im[ip0] * v->c3.c3re[ip1];
}
#pragma omp end declare target

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
#pragma omp declare target
void fsu3matdagxsu3matdag(
    const su3_mat_field *u, const su3_mat_field *v, su3_dble *res, int n, int i)
{
   int ip2 = plaq_uidx2(n, i);
   int ip3 = plaq_uidx3(n, i);

   // printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip2, ip3);

   res->c11.re = u->c1.c1re[ip2] * v->c1.c1re[ip3] + u->c1.c1im[ip2] * -v->c1.c1im[ip3] +
                 u->c2.c1re[ip2] * v->c1.c2re[ip3] + u->c2.c1im[ip2] * -v->c1.c2im[ip3] +
                 u->c3.c1re[ip2] * v->c1.c3re[ip3] + u->c3.c1im[ip2] * -v->c1.c3im[ip3];
   res->c11.im = u->c1.c1re[ip2] * -v->c1.c1im[ip3] - u->c1.c1im[ip2] * v->c1.c1re[ip3] +
                 u->c2.c1re[ip2] * -v->c1.c2im[ip3] - u->c2.c1im[ip2] * v->c1.c2re[ip3] +
                 u->c3.c1re[ip2] * -v->c1.c3im[ip3] - u->c3.c1im[ip2] * v->c1.c3re[ip3];
   res->c21.re = u->c1.c2re[ip2] * v->c1.c1re[ip3] + u->c1.c2im[ip2] * -v->c1.c1im[ip3] +
                 u->c2.c2re[ip2] * v->c1.c2re[ip3] + u->c2.c2im[ip2] * -v->c1.c2im[ip3] +
                 u->c3.c2re[ip2] * v->c1.c3re[ip3] + u->c3.c2im[ip2] * -v->c1.c3im[ip3];
   res->c21.im = u->c1.c2re[ip2] * -v->c1.c1im[ip3] - u->c1.c2im[ip2] * v->c1.c1re[ip3] +
                 u->c2.c2re[ip2] * -v->c1.c2im[ip3] - u->c2.c2im[ip2] * v->c1.c2re[ip3] +
                 u->c3.c2re[ip2] * -v->c1.c3im[ip3] - u->c3.c2im[ip2] * v->c1.c3re[ip3];
   res->c31.re = u->c1.c3re[ip2] * v->c1.c1re[ip3] + u->c1.c3im[ip2] * -v->c1.c1im[ip3] +
                 u->c2.c3re[ip2] * v->c1.c2re[ip3] + u->c2.c3im[ip2] * -v->c1.c2im[ip3] +
                 u->c3.c3re[ip2] * v->c1.c3re[ip3] + u->c3.c3im[ip2] * -v->c1.c3im[ip3];
   res->c31.im = u->c1.c3re[ip2] * -v->c1.c1im[ip3] - u->c1.c3im[ip2] * v->c1.c1re[ip3] +
                 u->c2.c3re[ip2] * -v->c1.c2im[ip3] - u->c2.c3im[ip2] * v->c1.c2re[ip3] +
                 u->c3.c3re[ip2] * -v->c1.c3im[ip3] - u->c3.c3im[ip2] * v->c1.c3re[ip3];

   res->c12.re = u->c1.c1re[ip2] * v->c2.c1re[ip3] + u->c1.c1im[ip2] * -v->c2.c1im[ip3] +
                 u->c2.c1re[ip2] * v->c2.c2re[ip3] + u->c2.c1im[ip2] * -v->c2.c2im[ip3] +
                 u->c3.c1re[ip2] * v->c2.c3re[ip3] + u->c3.c1im[ip2] * -v->c2.c3im[ip3];
   res->c12.im = u->c1.c1re[ip2] * -v->c2.c1im[ip3] - u->c1.c1im[ip2] * v->c2.c1re[ip3] +
                 u->c2.c1re[ip2] * -v->c2.c2im[ip3] - u->c2.c1im[ip2] * v->c2.c2re[ip3] +
                 u->c3.c1re[ip2] * -v->c2.c3im[ip3] - u->c3.c1im[ip2] * v->c2.c3re[ip3];
   res->c22.re = u->c1.c2re[ip2] * v->c2.c1re[ip3] + u->c1.c2im[ip2] * -v->c2.c1im[ip3] +
                 u->c2.c2re[ip2] * v->c2.c2re[ip3] + u->c2.c2im[ip2] * -v->c2.c2im[ip3] +
                 u->c3.c2re[ip2] * v->c2.c3re[ip3] + u->c3.c2im[ip2] * -v->c2.c3im[ip3];
   res->c22.im = u->c1.c2re[ip2] * -v->c2.c1im[ip3] - u->c1.c2im[ip2] * v->c2.c1re[ip3] +
                 u->c2.c2re[ip2] * -v->c2.c2im[ip3] - u->c2.c2im[ip2] * v->c2.c2re[ip3] +
                 u->c3.c2re[ip2] * -v->c2.c3im[ip3] - u->c3.c2im[ip2] * v->c2.c3re[ip3];
   res->c32.re = u->c1.c3re[ip2] * v->c2.c1re[ip3] + u->c1.c3im[ip2] * -v->c2.c1im[ip3] +
                 u->c2.c3re[ip2] * v->c2.c2re[ip3] + u->c2.c3im[ip2] * -v->c2.c2im[ip3] +
                 u->c3.c3re[ip2] * v->c2.c3re[ip3] + u->c3.c3im[ip2] * -v->c2.c3im[ip3];
   res->c32.im = u->c1.c3re[ip2] * -v->c2.c1im[ip3] - u->c1.c3im[ip2] * v->c2.c1re[ip3] +
                 u->c2.c3re[ip2] * -v->c2.c2im[ip3] - u->c2.c3im[ip2] * v->c2.c2re[ip3] +
                 u->c3.c3re[ip2] * -v->c2.c3im[ip3] - u->c3.c3im[ip2] * v->c2.c3re[ip3];

   res->c13.re = u->c1.c1re[ip2] * v->c3.c1re[ip3] + u->c1.c1im[ip2] * -v->c3.c1im[ip3] +
                 u->c2.c1re[ip2] * v->c3.c2re[ip3] + u->c2.c1im[ip2] * -v->c3.c2im[ip3] +
                 u->c3.c1re[ip2] * v->c3.c3re[ip3] + u->c3.c1im[ip2] * -v->c3.c3im[ip3];
   res->c13.im = u->c1.c1re[ip2] * -v->c3.c1im[ip3] - u->c1.c1im[ip2] * v->c3.c1re[ip3] +
                 u->c2.c1re[ip2] * -v->c3.c2im[ip3] - u->c2.c1im[ip2] * v->c3.c2re[ip3] +
                 u->c3.c1re[ip2] * -v->c3.c3im[ip3] - u->c3.c1im[ip2] * v->c3.c3re[ip3];
   res->c23.re = u->c1.c2re[ip2] * v->c3.c1re[ip3] + u->c1.c2im[ip2] * -v->c3.c1im[ip3] +
                 u->c2.c2re[ip2] * v->c3.c2re[ip3] + u->c2.c2im[ip2] * -v->c3.c2im[ip3] +
                 u->c3.c2re[ip2] * v->c3.c3re[ip3] + u->c3.c2im[ip2] * -v->c3.c3im[ip3];
   res->c23.im = u->c1.c2re[ip2] * -v->c3.c1im[ip3] - u->c1.c2im[ip2] * v->c3.c1re[ip3] +
                 u->c2.c2re[ip2] * -v->c3.c2im[ip3] - u->c2.c2im[ip2] * v->c3.c2re[ip3] +
                 u->c3.c2re[ip2] * -v->c3.c3im[ip3] - u->c3.c2im[ip2] * v->c3.c3re[ip3];
   res->c33.re = u->c1.c3re[ip2] * v->c3.c1re[ip3] + u->c1.c3im[ip2] * -v->c3.c1im[ip3] +
                 u->c2.c3re[ip2] * v->c3.c2re[ip3] + u->c2.c3im[ip2] * -v->c3.c2im[ip3] +
                 u->c3.c3re[ip2] * v->c3.c3re[ip3] + u->c3.c3im[ip2] * -v->c3.c3im[ip3];
   res->c33.im = u->c1.c3re[ip2] * -v->c3.c1im[ip3] - u->c1.c3im[ip2] * v->c3.c1re[ip3] +
                 u->c2.c3re[ip2] * -v->c3.c2im[ip3] - u->c2.c3im[ip2] * v->c3.c2re[ip3] +
                 u->c3.c3re[ip2] * -v->c3.c3im[ip3] - u->c3.c3im[ip2] * v->c3.c3re[ip3];
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

   /* ----------------------------------------------------------------
    * Diagonal element (1,1): row 1 of A · col 1 of B
    *
    * A_1k = sum over row 1 of (u*v):
    *   A_1k = u_row1 · v_colk  (standard mat mul)
    *
    * B_k1 = element (k,1) of (u^dag * v^dag):
    *   (u^dag)_mk = conj(u_km), (v^dag)_kn = conj(v_nk)
    *   B_k1 = sum_m conj(u_mk)[ip2] * conj(v_1m)[ip3]
    *
    * We load only 6 doubles from u (row 1) and 6 from v (col 1)
    * for A, then 6 from u (col 1) and 6 from v (row 1) for B,
    * keeping them in scoped blocks so registers are freed after each.
    * ---------------------------------------------------------------- */

   /* --- Diagonal (1,1) --- */
   {
      /* Row 1 of A = u_row1 * v  (forward product, row 1) */
      double u11r = u->c1.c1re[ip0], u11i = u->c1.c1im[ip0];
      double u12r = u->c1.c2re[ip0], u12i = u->c1.c2im[ip0];
      double u13r = u->c1.c3re[ip0], u13i = u->c1.c3im[ip0];

      double v11r = u->c1.c1re[ip1], v11i = u->c1.c1im[ip1];
      double v21r = u->c2.c1re[ip1], v21i = u->c2.c1im[ip1];
      double v31r = u->c3.c1re[ip1], v31i = u->c3.c1im[ip1];

      /* A_11 = u_row1 · v_col1 */
      double a11r = u11r*v11r - u11i*v11i + u12r*v21r - u12i*v21i + u13r*v31r - u13i*v31i;
      double a11i = u11r*v11i + u11i*v11r + u12r*v21i + u12i*v21r + u13r*v31i + u13i*v31r;

      double v12r = u->c1.c2re[ip1], v12i = u->c1.c2im[ip1];
      double v22r = u->c2.c2re[ip1], v22i = u->c2.c2im[ip1];
      double v32r = u->c3.c2re[ip1], v32i = u->c3.c2im[ip1];

      /* A_12 */
      double a12r = u11r*v12r - u11i*v12i + u12r*v22r - u12i*v22i + u13r*v32r - u13i*v32i;
      double a12i = u11r*v12i + u11i*v12r + u12r*v22i + u12i*v22r + u13r*v32i + u13i*v32r;

      double v13r = u->c1.c3re[ip1], v13i = u->c1.c3im[ip1];
      double v23r = u->c2.c3re[ip1], v23i = u->c2.c3im[ip1];
      double v33r = u->c3.c3re[ip1], v33i = u->c3.c3im[ip1];

      /* A_13 */
      double a13r = u11r*v13r - u11i*v13i + u12r*v23r - u12i*v23i + u13r*v33r - u13i*v33i;
      double a13i = u11r*v13i + u11i*v13r + u12r*v23i + u12i*v23r + u13r*v33i + u13i*v33r;

      /* Col 1 of B = (u^dag * v^dag)_k1
       * (u^dag)_1k = conj(u_k1)[ip2], (v^dag)_k1 = conj(v_1k)[ip3]
       * B_11 = sum_m conj(u_m1)[ip2] * conj(v_1m)[ip3] */
      double ub11r = u->c1.c1re[ip2], ub11i = u->c1.c1im[ip2];
      double ub21r = u->c2.c1re[ip2], ub21i = u->c2.c1im[ip2];
      double ub31r = u->c3.c1re[ip2], ub31i = u->c3.c1im[ip2];

      double vb11r = u->c1.c1re[ip3], vb11i = u->c1.c1im[ip3];
      double vb12r = u->c1.c2re[ip3], vb12i = u->c1.c2im[ip3];
      double vb13r = u->c1.c3re[ip3], vb13i = u->c1.c3im[ip3];

      /* B_11 = conj(u_col1) · conj(v_row1) */
      double b11r =  ub11r*vb11r - ub11i*(-vb11i) + ub21r*vb12r - ub21i*(-vb12i) + ub31r*vb13r - ub31i*(-vb13i);
      double b11i = -ub11r*vb11i - ub11i*vb11r   - ub21r*vb12i - ub21i*vb12r    - ub31r*vb13i - ub31i*vb13r;

      double ub12r = u->c1.c2re[ip2], ub12i = u->c1.c2im[ip2];
      double ub22r = u->c2.c2re[ip2], ub22i = u->c2.c2im[ip2];
      double ub32r = u->c3.c2re[ip2], ub32i = u->c3.c2im[ip2];

      double vb21r = u->c2.c1re[ip3], vb21i = u->c2.c1im[ip3];
      double vb22r = u->c2.c2re[ip3], vb22i = u->c2.c2im[ip3];
      double vb23r = u->c2.c3re[ip3], vb23i = u->c2.c3im[ip3];

      double b21r =  ub12r*vb21r - ub12i*(-vb21i) + ub22r*vb22r - ub22i*(-vb22i) + ub32r*vb23r - ub32i*(-vb23i);
      double b21i = -ub12r*vb21i - ub12i*vb21r    - ub22r*vb22i - ub22i*vb22r    - ub32r*vb23i - ub32i*vb23r;

      double ub13r = u->c1.c3re[ip2], ub13i = u->c1.c3im[ip2];
      double ub23r = u->c2.c3re[ip2], ub23i = u->c2.c3im[ip2];
      double ub33r = u->c3.c3re[ip2], ub33i = u->c3.c3im[ip2];

      double vb31r = u->c3.c1re[ip3], vb31i = u->c3.c1im[ip3];
      double vb32r = u->c3.c2re[ip3], vb32i = u->c3.c2im[ip3];
      double vb33r = u->c3.c3re[ip3], vb33i = u->c3.c3im[ip3];

      double b31r =  ub13r*vb31r - ub13i*(-vb31i) + ub23r*vb32r - ub23i*(-vb32i) + ub33r*vb33r - ub33i*(-vb33i);
      double b31i = -ub13r*vb31i - ub13i*vb31r    - ub23r*vb32i - ub23i*vb32r    - ub33r*vb33i - ub33i*vb33r;

      /* Re Tr contribution from row 1: Re(A_11*B_11 + A_12*B_21 + A_13*B_31) */
      tr += a11r*b11r - a11i*b11i
          + a12r*b21r - a12i*b21i
          + a13r*b31r - a13i*b31i;
   } /* all row-1 registers freed here */

   /* --- Diagonal (2,2) --- */
   {
      double u21r = u->c2.c1re[ip0], u21i = u->c2.c1im[ip0];
      double u22r = u->c2.c2re[ip0], u22i = u->c2.c2im[ip0];
      double u23r = u->c2.c3re[ip0], u23i = u->c2.c3im[ip0];

      double v11r = u->c1.c1re[ip1], v11i = u->c1.c1im[ip1];
      double v21r = u->c2.c1re[ip1], v21i = u->c2.c1im[ip1];
      double v31r = u->c3.c1re[ip1], v31i = u->c3.c1im[ip1];

      double a21r = u21r*v11r - u21i*v11i + u22r*v21r - u22i*v21i + u23r*v31r - u23i*v31i;
      double a21i = u21r*v11i + u21i*v11r + u22r*v21i + u22i*v21r + u23r*v31i + u23i*v31r;

      double v12r = u->c1.c2re[ip1], v12i = u->c1.c2im[ip1];
      double v22r = u->c2.c2re[ip1], v22i = u->c2.c2im[ip1];
      double v32r = u->c3.c2re[ip1], v32i = u->c3.c2im[ip1];

      double a22r = u21r*v12r - u21i*v12i + u22r*v22r - u22i*v22i + u23r*v32r - u23i*v32i;
      double a22i = u21r*v12i + u21i*v12r + u22r*v22i + u22i*v22r + u23r*v32i + u23i*v32r;

      double v13r = u->c1.c3re[ip1], v13i = u->c1.c3im[ip1];
      double v23r = u->c2.c3re[ip1], v23i = u->c2.c3im[ip1];
      double v33r = u->c3.c3re[ip1], v33i = u->c3.c3im[ip1];

      double a23r = u21r*v13r - u21i*v13i + u22r*v23r - u22i*v23i + u23r*v33r - u23i*v33i;
      double a23i = u21r*v13i + u21i*v13r + u22r*v23i + u22i*v23r + u23r*v33i + u23i*v33r;

      double ub11r = u->c1.c1re[ip2], ub11i = u->c1.c1im[ip2];
      double ub21r = u->c2.c1re[ip2], ub21i = u->c2.c1im[ip2];
      double ub31r = u->c3.c1re[ip2], ub31i = u->c3.c1im[ip2];

      double vb11r = u->c1.c1re[ip3], vb11i = u->c1.c1im[ip3];
      double vb12r = u->c1.c2re[ip3], vb12i = u->c1.c2im[ip3];
      double vb13r = u->c1.c3re[ip3], vb13i = u->c1.c3im[ip3];

      double b12r =  ub11r*vb11r - ub11i*(-vb11i) + ub21r*vb12r - ub21i*(-vb12i) + ub31r*vb13r - ub31i*(-vb13i);
      double b12i = -ub11r*vb11i - ub11i*vb11r    - ub21r*vb12i - ub21i*vb12r    - ub31r*vb13i - ub31i*vb13r;

      double ub12r = u->c1.c2re[ip2], ub12i = u->c1.c2im[ip2];
      double ub22r = u->c2.c2re[ip2], ub22i = u->c2.c2im[ip2];
      double ub32r = u->c3.c2re[ip2], ub32i = u->c3.c2im[ip2];

      double vb21r = u->c2.c1re[ip3], vb21i = u->c2.c1im[ip3];
      double vb22r = u->c2.c2re[ip3], vb22i = u->c2.c2im[ip3];
      double vb23r = u->c2.c3re[ip3], vb23i = u->c2.c3im[ip3];

      double b22r =  ub12r*vb21r - ub12i*(-vb21i) + ub22r*vb22r - ub22i*(-vb22i) + ub32r*vb23r - ub32i*(-vb23i);
      double b22i = -ub12r*vb21i - ub12i*vb21r    - ub22r*vb22i - ub22i*vb22r    - ub32r*vb23i - ub32i*vb23r;

      double ub13r = u->c1.c3re[ip2], ub13i = u->c1.c3im[ip2];
      double ub23r = u->c2.c3re[ip2], ub23i = u->c2.c3im[ip2];
      double ub33r = u->c3.c3re[ip2], ub33i = u->c3.c3im[ip2];

      double vb31r = u->c3.c1re[ip3], vb31i = u->c3.c1im[ip3];
      double vb32r = u->c3.c2re[ip3], vb32i = u->c3.c2im[ip3];
      double vb33r = u->c3.c3re[ip3], vb33i = u->c3.c3im[ip3];

      double b32r =  ub13r*vb31r - ub13i*(-vb31i) + ub23r*vb32r - ub23i*(-vb32i) + ub33r*vb33r - ub33i*(-vb33i);
      double b32i = -ub13r*vb31i - ub13i*vb31r    - ub23r*vb32i - ub23i*vb32r    - ub33r*vb33i - ub33i*vb33r;

      tr += a21r*b12r - a21i*b12i
          + a22r*b22r - a22i*b22i
          + a23r*b32r - a23i*b32i;
   } /* all row-2 registers freed here */

   /* --- Diagonal (3,3) --- */
   {
      double u31r = u->c3.c1re[ip0], u31i = u->c3.c1im[ip0];
      double u32r = u->c3.c2re[ip0], u32i = u->c3.c2im[ip0];
      double u33r = u->c3.c3re[ip0], u33i = u->c3.c3im[ip0];

      double v11r = u->c1.c1re[ip1], v11i = u->c1.c1im[ip1];
      double v21r = u->c2.c1re[ip1], v21i = u->c2.c1im[ip1];
      double v31r = u->c3.c1re[ip1], v31i = u->c3.c1im[ip1];

      double a31r = u31r*v11r - u31i*v11i + u32r*v21r - u32i*v21i + u33r*v31r - u33i*v31i;
      double a31i = u31r*v11i + u31i*v11r + u32r*v21i + u32i*v21r + u33r*v31i + u33i*v31r;

      double v12r = u->c1.c2re[ip1], v12i = u->c1.c2im[ip1];
      double v22r = u->c2.c2re[ip1], v22i = u->c2.c2im[ip1];
      double v32r = u->c3.c2re[ip1], v32i = u->c3.c2im[ip1];

      double a32r = u31r*v12r - u31i*v12i + u32r*v22r - u32i*v22i + u33r*v32r - u33i*v32i;
      double a32i = u31r*v12i + u31i*v12r + u32r*v22i + u32i*v22r + u33r*v32i + u33i*v32r;

      double v13r = u->c1.c3re[ip1], v13i = u->c1.c3im[ip1];
      double v23r = u->c2.c3re[ip1], v23i = u->c2.c3im[ip1];
      double v33r = u->c3.c3re[ip1], v33i = u->c3.c3im[ip1];

      double a33r = u31r*v13r - u31i*v13i + u32r*v23r - u32i*v23i + u33r*v33r - u33i*v33i;
      double a33i = u31r*v13i + u31i*v13r + u32r*v23i + u32i*v23r + u33r*v33i + u33i*v33r;

      double ub11r = u->c1.c1re[ip2], ub11i = u->c1.c1im[ip2];
      double ub21r = u->c2.c1re[ip2], ub21i = u->c2.c1im[ip2];
      double ub31r = u->c3.c1re[ip2], ub31i = u->c3.c1im[ip2];

      double vb11r = u->c1.c1re[ip3], vb11i = u->c1.c1im[ip3];
      double vb12r = u->c1.c2re[ip3], vb12i = u->c1.c2im[ip3];
      double vb13r = u->c1.c3re[ip3], vb13i = u->c1.c3im[ip3];

      double b13r =  ub11r*vb11r - ub11i*(-vb11i) + ub21r*vb12r - ub21i*(-vb12i) + ub31r*vb13r - ub31i*(-vb13i);
      double b13i = -ub11r*vb11i - ub11i*vb11r    - ub21r*vb12i - ub21i*vb12r    - ub31r*vb13i - ub31i*vb13r;

      double ub12r = u->c1.c2re[ip2], ub12i = u->c1.c2im[ip2];
      double ub22r = u->c2.c2re[ip2], ub22i = u->c2.c2im[ip2];
      double ub32r = u->c3.c2re[ip2], ub32i = u->c3.c2im[ip2];

      double vb21r = u->c2.c1re[ip3], vb21i = u->c2.c1im[ip3];
      double vb22r = u->c2.c2re[ip3], vb22i = u->c2.c2im[ip3];
      double vb23r = u->c2.c3re[ip3], vb23i = u->c2.c3im[ip3];

      double b23r =  ub12r*vb21r - ub12i*(-vb21i) + ub22r*vb22r - ub22i*(-vb22i) + ub32r*vb23r - ub32i*(-vb23i);
      double b23i = -ub12r*vb21i - ub12i*vb21r    - ub22r*vb22i - ub22i*vb22r    - ub32r*vb23i - ub32i*vb23r;

      double ub13r = u->c1.c3re[ip2], ub13i = u->c1.c3im[ip2];
      double ub23r = u->c2.c3re[ip2], ub23i = u->c2.c3im[ip2];
      double ub33r = u->c3.c3re[ip2], ub33i = u->c3.c3im[ip2];

      double vb31r = u->c3.c1re[ip3], vb31i = u->c3.c1im[ip3];
      double vb32r = u->c3.c2re[ip3], vb32i = u->c3.c2im[ip3];
      double vb33r = u->c3.c3re[ip3], vb33i = u->c3.c3im[ip3];

      double b33r =  ub13r*vb31r - ub13i*(-vb31i) + ub23r*vb32r - ub23i*(-vb32i) + ub33r*vb33r - ub33i*(-vb33i);
      double b33i = -ub13r*vb31i - ub13i*vb31r    - ub23r*vb32i - ub23i*vb32r    - ub33r*vb33i - ub33i*vb33r;

      tr += a31r*b13r - a31i*b13i
          + a32r*b23r - a32i*b23i
          + a33r*b33r - a33i*b33i;
   } /* all row-3 registers freed here */

   return tr;
}
#pragma omp end declare target

#endif
