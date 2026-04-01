
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

// #pragma omp declare target
// double fsu3matxsu3mat_retrace(
//     const su3_mat_field *u, const su3_mat_field *v, int i)
// {
//    double tr_1 = 0.0;
//    double tr_2 = 0.0;
//    double tr_3 = 0.0;
// 
//    tr_1 += u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i];
//    tr_1 += u->c1.c2re[i] * v->c2.c1re[i] - u->c1.c2im[i] * v->c2.c1im[i];
//    tr_1 += u->c1.c3re[i] * v->c3.c1re[i] - u->c1.c3im[i] * v->c3.c1im[i];
// 
//    tr_2 += u->c2.c1re[i] * v->c1.c2re[i] - u->c2.c1im[i] * v->c1.c2im[i];
//    tr_2 += u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i];
//    tr_2 += u->c2.c3re[i] * v->c3.c2re[i] - u->c2.c3im[i] * v->c3.c2im[i];
// 
//    tr_3 += u->c3.c1re[i] * v->c1.c3re[i] - u->c3.c1im[i] * v->c1.c3im[i];
//    tr_3 += u->c3.c2re[i] * v->c2.c3re[i] - u->c3.c2im[i] * v->c2.c3im[i];
//    tr_3 += u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];
// 
//    return tr_1 + tr_2 + tr_3;
// }
// #pragma omp end declare target

// #pragma omp declare target
// void fsu3matxsu3mat(
//     const su3_mat_field *u, su3_dble *res, int ip0, int ip1)
// {
//    //  printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip0, ip1);
// 
//    res->c11.re = u->c1.c1re[ip0] * u->c1.c1re[ip1] - u->c1.c1im[ip0] * u->c1.c1im[ip1] +
//                  u->c1.c2re[ip0] * u->c1.c2re[ip1] - u->c1.c2im[ip0] * u->c1.c2im[ip1] +
//                  u->c1.c3re[ip0] * u->c1.c3re[ip1] - u->c1.c3im[ip0] * u->c1.c3im[ip1];
//    res->c11.im = u->c1.c1re[ip0] * u->c1.c1im[ip1] + u->c1.c1im[ip0] * u->c1.c1re[ip1] +
//                  u->c1.c2re[ip0] * u->c1.c2im[ip1] + u->c1.c2im[ip0] * u->c1.c2re[ip1] +
//                  u->c1.c3re[ip0] * u->c1.c3im[ip1] + u->c1.c3im[ip0] * u->c1.c3re[ip1];
//    res->c12.re = u->c2.c1re[ip0] * u->c1.c1re[ip1] - u->c2.c1im[ip0] * u->c1.c1im[ip1] +
//                  u->c2.c2re[ip0] * u->c1.c2re[ip1] - u->c2.c2im[ip0] * u->c1.c2im[ip1] +
//                  u->c2.c3re[ip0] * u->c1.c3re[ip1] - u->c2.c3im[ip0] * u->c1.c3im[ip1];
//    res->c12.im = u->c2.c1re[ip0] * u->c1.c1im[ip1] + u->c2.c1im[ip0] * u->c1.c1re[ip1] +
//                  u->c2.c2re[ip0] * u->c1.c2im[ip1] + u->c2.c2im[ip0] * u->c1.c2re[ip1] +
//                  u->c2.c3re[ip0] * u->c1.c3im[ip1] + u->c2.c3im[ip0] * u->c1.c3re[ip1];
//    res->c13.re = u->c3.c1re[ip0] * u->c1.c1re[ip1] - u->c3.c1im[ip0] * u->c1.c1im[ip1] +
//                  u->c3.c2re[ip0] * u->c1.c2re[ip1] - u->c3.c2im[ip0] * u->c1.c2im[ip1] +
//                  u->c3.c3re[ip0] * u->c1.c3re[ip1] - u->c3.c3im[ip0] * u->c1.c3im[ip1];
//    res->c13.im = u->c3.c1re[ip0] * u->c1.c1im[ip1] + u->c3.c1im[ip0] * u->c1.c1re[ip1] +
//                  u->c3.c2re[ip0] * u->c1.c2im[ip1] + u->c3.c2im[ip0] * u->c1.c2re[ip1] +
//                  u->c3.c3re[ip0] * u->c1.c3im[ip1] + u->c3.c3im[ip0] * u->c1.c3re[ip1];
// 
//    res->c21.re = u->c1.c1re[ip0] * u->c2.c1re[ip1] - u->c1.c1im[ip0] * u->c2.c1im[ip1] +
//                  u->c1.c2re[ip0] * u->c2.c2re[ip1] - u->c1.c2im[ip0] * u->c2.c2im[ip1] +
//                  u->c1.c3re[ip0] * u->c2.c3re[ip1] - u->c1.c3im[ip0] * u->c2.c3im[ip1];
//    res->c21.im = u->c1.c1re[ip0] * u->c2.c1im[ip1] + u->c1.c1im[ip0] * u->c2.c1re[ip1] +
//                  u->c1.c2re[ip0] * u->c2.c2im[ip1] + u->c1.c2im[ip0] * u->c2.c2re[ip1] +
//                  u->c1.c3re[ip0] * u->c2.c3im[ip1] + u->c1.c3im[ip0] * u->c2.c3re[ip1];
//    res->c22.re = u->c2.c1re[ip0] * u->c2.c1re[ip1] - u->c2.c1im[ip0] * u->c2.c1im[ip1] +
//                  u->c2.c2re[ip0] * u->c2.c2re[ip1] - u->c2.c2im[ip0] * u->c2.c2im[ip1] +
//                  u->c2.c3re[ip0] * u->c2.c3re[ip1] - u->c2.c3im[ip0] * u->c2.c3im[ip1];
//    res->c22.im = u->c2.c1re[ip0] * u->c2.c1im[ip1] + u->c2.c1im[ip0] * u->c2.c1re[ip1] +
//                  u->c2.c2re[ip0] * u->c2.c2im[ip1] + u->c2.c2im[ip0] * u->c2.c2re[ip1] +
//                  u->c2.c3re[ip0] * u->c2.c3im[ip1] + u->c2.c3im[ip0] * u->c2.c3re[ip1];
//    res->c23.re = u->c3.c1re[ip0] * u->c2.c1re[ip1] - u->c3.c1im[ip0] * u->c2.c1im[ip1] +
//                  u->c3.c2re[ip0] * u->c2.c2re[ip1] - u->c3.c2im[ip0] * u->c2.c2im[ip1] +
//                  u->c3.c3re[ip0] * u->c2.c3re[ip1] - u->c3.c3im[ip0] * u->c2.c3im[ip1];
//    res->c23.im = u->c3.c1re[ip0] * u->c2.c1im[ip1] + u->c3.c1im[ip0] * u->c2.c1re[ip1] +
//                  u->c3.c2re[ip0] * u->c2.c2im[ip1] + u->c3.c2im[ip0] * u->c2.c2re[ip1] +
//                  u->c3.c3re[ip0] * u->c2.c3im[ip1] + u->c3.c3im[ip0] * u->c2.c3re[ip1];
// 
//    res->c31.re = u->c1.c1re[ip0] * u->c3.c1re[ip1] - u->c1.c1im[ip0] * u->c3.c1im[ip1] +
//                  u->c1.c2re[ip0] * u->c3.c2re[ip1] - u->c1.c2im[ip0] * u->c3.c2im[ip1] +
//                  u->c1.c3re[ip0] * u->c3.c3re[ip1] - u->c1.c3im[ip0] * u->c3.c3im[ip1];
//    res->c31.im = u->c1.c1re[ip0] * u->c3.c1im[ip1] + u->c1.c1im[ip0] * u->c3.c1re[ip1] +
//                  u->c1.c2re[ip0] * u->c3.c2im[ip1] + u->c1.c2im[ip0] * u->c3.c2re[ip1] +
//                  u->c1.c3re[ip0] * u->c3.c3im[ip1] + u->c1.c3im[ip0] * u->c3.c3re[ip1];
//    res->c32.re = u->c2.c1re[ip0] * u->c3.c1re[ip1] - u->c2.c1im[ip0] * u->c3.c1im[ip1] +
//                  u->c2.c2re[ip0] * u->c3.c2re[ip1] - u->c2.c2im[ip0] * u->c3.c2im[ip1] +
//                  u->c2.c3re[ip0] * u->c3.c3re[ip1] - u->c2.c3im[ip0] * u->c3.c3im[ip1];
//    res->c32.im = u->c2.c1re[ip0] * u->c3.c1im[ip1] + u->c2.c1im[ip0] * u->c3.c1re[ip1] +
//                  u->c2.c2re[ip0] * u->c3.c2im[ip1] + u->c2.c2im[ip0] * u->c3.c2re[ip1] +
//                  u->c2.c3re[ip0] * u->c3.c3im[ip1] + u->c2.c3im[ip0] * u->c3.c3re[ip1];
//    res->c33.re = u->c3.c1re[ip0] * u->c3.c1re[ip1] - u->c3.c1im[ip0] * u->c3.c1im[ip1] +
//                  u->c3.c2re[ip0] * u->c3.c2re[ip1] - u->c3.c2im[ip0] * u->c3.c2im[ip1] +
//                  u->c3.c3re[ip0] * u->c3.c3re[ip1] - u->c3.c3im[ip0] * u->c3.c3im[ip1];
//    res->c33.im = u->c3.c1re[ip0] * u->c3.c1im[ip1] + u->c3.c1im[ip0] * u->c3.c1re[ip1] +
//                  u->c3.c2re[ip0] * u->c3.c2im[ip1] + u->c3.c2im[ip0] * u->c3.c2re[ip1] +
//                  u->c3.c3re[ip0] * u->c3.c3im[ip1] + u->c3.c3im[ip0] * u->c3.c3re[ip1];
// }
// #pragma omp end declare target

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
// #pragma omp declare target
// void fsu3matdagxsu3matdag(
//     const su3_mat_field *u, su3_dble *res, int ip2, int ip3)
// {
//    // printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip2, ip3);
// 
//    res->c11.re = u->c1.c1re[ip2] * u->c1.c1re[ip3] + u->c1.c1im[ip2] * -u->c1.c1im[ip3] +
//                  u->c2.c1re[ip2] * u->c1.c2re[ip3] + u->c2.c1im[ip2] * -u->c1.c2im[ip3] +
//                  u->c3.c1re[ip2] * u->c1.c3re[ip3] + u->c3.c1im[ip2] * -u->c1.c3im[ip3];
//    res->c11.im = u->c1.c1re[ip2] * -u->c1.c1im[ip3] - u->c1.c1im[ip2] * u->c1.c1re[ip3] +
//                  u->c2.c1re[ip2] * -u->c1.c2im[ip3] - u->c2.c1im[ip2] * u->c1.c2re[ip3] +
//                  u->c3.c1re[ip2] * -u->c1.c3im[ip3] - u->c3.c1im[ip2] * u->c1.c3re[ip3];
//    res->c21.re = u->c1.c2re[ip2] * u->c1.c1re[ip3] + u->c1.c2im[ip2] * -u->c1.c1im[ip3] +
//                  u->c2.c2re[ip2] * u->c1.c2re[ip3] + u->c2.c2im[ip2] * -u->c1.c2im[ip3] +
//                  u->c3.c2re[ip2] * u->c1.c3re[ip3] + u->c3.c2im[ip2] * -u->c1.c3im[ip3];
//    res->c21.im = u->c1.c2re[ip2] * -u->c1.c1im[ip3] - u->c1.c2im[ip2] * u->c1.c1re[ip3] +
//                  u->c2.c2re[ip2] * -u->c1.c2im[ip3] - u->c2.c2im[ip2] * u->c1.c2re[ip3] +
//                  u->c3.c2re[ip2] * -u->c1.c3im[ip3] - u->c3.c2im[ip2] * u->c1.c3re[ip3];
//    res->c31.re = u->c1.c3re[ip2] * u->c1.c1re[ip3] + u->c1.c3im[ip2] * -u->c1.c1im[ip3] +
//                  u->c2.c3re[ip2] * u->c1.c2re[ip3] + u->c2.c3im[ip2] * -u->c1.c2im[ip3] +
//                  u->c3.c3re[ip2] * u->c1.c3re[ip3] + u->c3.c3im[ip2] * -u->c1.c3im[ip3];
//    res->c31.im = u->c1.c3re[ip2] * -u->c1.c1im[ip3] - u->c1.c3im[ip2] * u->c1.c1re[ip3] +
//                  u->c2.c3re[ip2] * -u->c1.c2im[ip3] - u->c2.c3im[ip2] * u->c1.c2re[ip3] +
//                  u->c3.c3re[ip2] * -u->c1.c3im[ip3] - u->c3.c3im[ip2] * u->c1.c3re[ip3];
// 
//    res->c12.re = u->c1.c1re[ip2] * u->c2.c1re[ip3] + u->c1.c1im[ip2] * -u->c2.c1im[ip3] +
//                  u->c2.c1re[ip2] * u->c2.c2re[ip3] + u->c2.c1im[ip2] * -u->c2.c2im[ip3] +
//                  u->c3.c1re[ip2] * u->c2.c3re[ip3] + u->c3.c1im[ip2] * -u->c2.c3im[ip3];
//    res->c12.im = u->c1.c1re[ip2] * -u->c2.c1im[ip3] - u->c1.c1im[ip2] * u->c2.c1re[ip3] +
//                  u->c2.c1re[ip2] * -u->c2.c2im[ip3] - u->c2.c1im[ip2] * u->c2.c2re[ip3] +
//                  u->c3.c1re[ip2] * -u->c2.c3im[ip3] - u->c3.c1im[ip2] * u->c2.c3re[ip3];
//    res->c22.re = u->c1.c2re[ip2] * u->c2.c1re[ip3] + u->c1.c2im[ip2] * -u->c2.c1im[ip3] +
//                  u->c2.c2re[ip2] * u->c2.c2re[ip3] + u->c2.c2im[ip2] * -u->c2.c2im[ip3] +
//                  u->c3.c2re[ip2] * u->c2.c3re[ip3] + u->c3.c2im[ip2] * -u->c2.c3im[ip3];
//    res->c22.im = u->c1.c2re[ip2] * -u->c2.c1im[ip3] - u->c1.c2im[ip2] * u->c2.c1re[ip3] +
//                  u->c2.c2re[ip2] * -u->c2.c2im[ip3] - u->c2.c2im[ip2] * u->c2.c2re[ip3] +
//                  u->c3.c2re[ip2] * -u->c2.c3im[ip3] - u->c3.c2im[ip2] * u->c2.c3re[ip3];
//    res->c32.re = u->c1.c3re[ip2] * u->c2.c1re[ip3] + u->c1.c3im[ip2] * -u->c2.c1im[ip3] +
//                  u->c2.c3re[ip2] * u->c2.c2re[ip3] + u->c2.c3im[ip2] * -u->c2.c2im[ip3] +
//                  u->c3.c3re[ip2] * u->c2.c3re[ip3] + u->c3.c3im[ip2] * -u->c2.c3im[ip3];
//    res->c32.im = u->c1.c3re[ip2] * -u->c2.c1im[ip3] - u->c1.c3im[ip2] * u->c2.c1re[ip3] +
//                  u->c2.c3re[ip2] * -u->c2.c2im[ip3] - u->c2.c3im[ip2] * u->c2.c2re[ip3] +
//                  u->c3.c3re[ip2] * -u->c2.c3im[ip3] - u->c3.c3im[ip2] * u->c2.c3re[ip3];
// 
//    res->c13.re = u->c1.c1re[ip2] * u->c3.c1re[ip3] + u->c1.c1im[ip2] * -u->c3.c1im[ip3] +
//                  u->c2.c1re[ip2] * u->c3.c2re[ip3] + u->c2.c1im[ip2] * -u->c3.c2im[ip3] +
//                  u->c3.c1re[ip2] * u->c3.c3re[ip3] + u->c3.c1im[ip2] * -u->c3.c3im[ip3];
//    res->c13.im = u->c1.c1re[ip2] * -u->c3.c1im[ip3] - u->c1.c1im[ip2] * u->c3.c1re[ip3] +
//                  u->c2.c1re[ip2] * -u->c3.c2im[ip3] - u->c2.c1im[ip2] * u->c3.c2re[ip3] +
//                  u->c3.c1re[ip2] * -u->c3.c3im[ip3] - u->c3.c1im[ip2] * u->c3.c3re[ip3];
//    res->c23.re = u->c1.c2re[ip2] * u->c3.c1re[ip3] + u->c1.c2im[ip2] * -u->c3.c1im[ip3] +
//                  u->c2.c2re[ip2] * u->c3.c2re[ip3] + u->c2.c2im[ip2] * -u->c3.c2im[ip3] +
//                  u->c3.c2re[ip2] * u->c3.c3re[ip3] + u->c3.c2im[ip2] * -u->c3.c3im[ip3];
//    res->c23.im = u->c1.c2re[ip2] * -u->c3.c1im[ip3] - u->c1.c2im[ip2] * u->c3.c1re[ip3] +
//                  u->c2.c2re[ip2] * -u->c3.c2im[ip3] - u->c2.c2im[ip2] * u->c3.c2re[ip3] +
//                  u->c3.c2re[ip2] * -u->c3.c3im[ip3] - u->c3.c2im[ip2] * u->c3.c3re[ip3];
//    res->c33.re = u->c1.c3re[ip2] * u->c3.c1re[ip3] + u->c1.c3im[ip2] * -u->c3.c1im[ip3] +
//                  u->c2.c3re[ip2] * u->c3.c2re[ip3] + u->c2.c3im[ip2] * -u->c3.c2im[ip3] +
//                  u->c3.c3re[ip2] * u->c3.c3re[ip3] + u->c3.c3im[ip2] * -u->c3.c3im[ip3];
//    res->c33.im = u->c1.c3re[ip2] * -u->c3.c1im[ip3] - u->c1.c3im[ip2] * u->c3.c1re[ip3] +
//                  u->c2.c3re[ip2] * -u->c3.c2im[ip3] - u->c2.c3im[ip2] * u->c3.c2re[ip3] +
//                  u->c3.c3re[ip2] * -u->c3.c3im[ip3] - u->c3.c3im[ip2] * u->c3.c3re[ip3];
// }
// #pragma omp end declare target

#pragma omp declare target
void fsu3matxsu3mat(
    const su3_mat_field *u, su3_dble *res, int ip0, int ip1)
{
   // printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip2, ip3);

   su3_vec_field psi;
   su3_vector_dble chi;

   psi=(*u).c1;
   fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c11=chi.c1;
   (*res).c21=chi.c2;
   (*res).c31=chi.c3;

   psi=(*u).c2;
   fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c12=chi.c1;
   (*res).c22=chi.c2;
   (*res).c32=chi.c3;

   psi=(*u).c3;
   fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c13=chi.c1;
   (*res).c23=chi.c2;
   (*res).c33=chi.c3;

}
#pragma omp end declare target


#pragma omp declare target
void fsu3matxsu3vec(const su3_mat_field *u, const su3_vec_field *v,
                   su3_vector_dble *res, int ip0, int ip1)
{
    // Restricted pointers, no ailiasing
    const complex * __restrict__ u11 = u->c1.c1;
    const complex * __restrict__ u12 = u->c1.c2;
    const complex * __restrict__ u13 = u->c1.c3;
    const complex * __restrict__ u21 = u->c2.c1;
    const complex * __restrict__ u22 = u->c2.c2;
    const complex * __restrict__ u23 = u->c2.c3;
    const complex * __restrict__ u31 = u->c3.c1;
    const complex * __restrict__ u32 = u->c3.c2;
    const complex * __restrict__ u33 = u->c3.c3;
    const complex * __restrict__ v1  = v->c1;
    const complex * __restrict__ v2  = v->c2;
    const complex * __restrict__ v3  = v->c3;

    // Each load fetches re and im together
    complex U11 = u11[ip0], U12 = u12[ip0], U13 = u13[ip0];
    complex U21 = u21[ip0], U22 = u22[ip0], U23 = u23[ip0];
    complex U31 = u31[ip0], U32 = u32[ip0], U33 = u33[ip0];
    complex V1  = v1[ip1],  V2  = v2[ip1],  V3  = v3[ip1];

    res->c1.re = U11.re*V1.re - U11.im*V1.im + U12.re*V2.re - U12.im*V2.im + U13.re*V3.re - U13.im*V3.im;
    res->c1.im = U11.re*V1.im + U11.im*V1.re + U12.re*V2.im + U12.im*V2.re + U13.re*V3.im + U13.im*V3.re;

    res->c2.re = U21.re*V1.re - U21.im*V1.im + U22.re*V2.re - U22.im*V2.im + U23.re*V3.re - U23.im*V3.im;
    res->c2.im = U21.re*V1.im + U21.im*V1.re + U22.re*V2.im + U22.im*V2.re + U23.re*V3.im + U23.im*V3.re;

    res->c3.re = U31.re*V1.re - U31.im*V1.im + U32.re*V2.re - U32.im*V2.im + U33.re*V3.re - U33.im*V3.im;
    res->c3.im = U31.re*V1.im + U31.im*V1.re + U32.re*V2.im + U32.im*V2.re + U33.re*V3.im + U33.im*V3.re;
}
#pragma omp end declare target

#endif

