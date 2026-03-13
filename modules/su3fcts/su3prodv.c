
/*******************************************************************************
*
* File su3prod.c
*
* Copyright (C) 2005, 2009-2013, 2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Products of double-precision 3x3 matrices.
*
*   void su3xsu3(su3_dble *u,su3_dble *v,su3_dble *w)
*     Computes w=u*v assuming that w is different from u.
*
*   void su3dagxsu3(su3_dble *u,su3_dble *v,su3_dble *w)
*     Computes w=u^dag*v assuming that w is different from u.
*
*   void su3xsu3dag(su3_dble *u,su3_dble *v,su3_dble *w)
*     Computes w=u*v^dag assuming that w is different from u and v.
*
*   void su3dagxsu3dag(su3_dble *u,su3_dble *v,su3_dble *w)
*     Computes w=u^dag*v^dag assuming that w is different from u and v.
*
*   void su3xu3alg(su3_dble *u,u3_alg_dble *X,su3_dble *v)
*     Computes v=u*X assuming that v is different from u.
*
*   void su3dagxu3alg(su3_dble *u,u3_alg_dble *X,su3_dble *v)
*     Computes v=u^dag*X assuming that v is different from u.
*
*   void u3algxsu3(u3_alg_dble *X,su3_dble *u,su3_dble *v)
*     Computes v=X*u assuming that v is different from u.
*
*   void u3algxsu3dag(u3_alg_dble *X,su3_dble *u,su3_dble *v)
*     Computes v=X*u^dag assuming that v is different from u.
*
*   double prod2su3alg(su3_dble *u,su3_dble *v,su3_alg_dble *X)
*     Computes the product w=u*v and assigns its traceless antihermitian
*     part (1/2)*[w-w^dag-(1/3)*tr{w-w^dag}] to X. The program returns
*     the real part of tr{w}.
*
*   void prod2u3alg(su3_dble *u,su3_dble *v,u3_alg_dble *X)
*     Computes the product w=u*v and assigns w-w^dag to X.
*
*   void rotate_su3alg(su3_dble *u,su3_alg_dble *X)
*     Replaces X by u*X*u^dag. The matrix u must be unitary but its
*     determinant may be different from 1.
*
*   void lieprod_u3alg(u3_alg_dble *X,u3_alg_dble *Y,u3_alg_dble *Z)
*     Assigns the commutator [X,Y] to Z.
*
* Unless stated otherwise, the matrices of type su3_dble are not assumed to
* be unitary or unimodular. They are just treated as general 3x3 complex
* matrices and the operations are applied to them as described.
*
* The elements X of the Lie algebra of U(3) are antihermitian 3x3 matrices
* that are represented by structures X with real entries X.c1,...,X.c9
* through
*
*  X_11=i*X.c1, X_22=i*X.c2, X_33=i*X.c3,
*
*  X_12=X.c4+i*X.c5, X_13=X.c6+i*X.c7, X_23=X.c8+i*X.c9
*
* The type su3_alg_dble [which represents elements of the Lie algebra of SU(3)]
* is described in the file linalg/liealg.c.
*
* The programs in this module do not perform any communications and are
* thread-safe.
*
* If SSE2 or AVX instructions are used, all su3_dble and su3_alg_dble matrices
* are assumed to be aligned to 16 byte boundaries.
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
void fsu3matxsu3mat_retrace(
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
void fsu3matxsu3mat(
   const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, const int n, const size_t i)
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
void fsu3matdagxsu3matdag(
   const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, const int n, const size_t i)
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