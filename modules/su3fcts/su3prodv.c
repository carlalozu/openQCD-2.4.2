
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
 *   void fsu3xsu3(
 *     const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, int n,
 *     int i)
 *   Computes w=u*v assuming that w is different from u.
 *
 *   void fsu3dagxsu3dag(
 *     const su3_mat_field *u, const su3_mat_field *v, su3_mat_field *res, int n,
 *     int i)
 *   Computes w=u^dag*v^dag assuming that w is different from u and v.
 *
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
static void _fsu3matconj_inverse_multiply(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
{
   res->c1.re = u->c1.c1re[ip0] * v->c1re[ip1] - u->c1.c1im[ip0] * v->c1im[ip1] +
                u->c2.c1re[ip0] * v->c2re[ip1] - u->c2.c1im[ip0] * v->c2im[ip1] +
                u->c3.c1re[ip0] * v->c3re[ip1] - u->c3.c1im[ip0] * v->c3im[ip1];
   res->c1.im = -u->c1.c1re[ip0] * v->c1im[ip1] - u->c1.c1im[ip0] * v->c1re[ip1] +
                -u->c2.c1re[ip0] * v->c2im[ip1] - u->c2.c1im[ip0] * v->c2re[ip1] +
                -u->c3.c1re[ip0] * v->c3im[ip1] - u->c3.c1im[ip0] * v->c3re[ip1];
   res->c2.re = u->c1.c2re[ip0] * v->c1re[ip1] - u->c1.c2im[ip0] * v->c1im[ip1] +
                u->c2.c2re[ip0] * v->c2re[ip1] - u->c2.c2im[ip0] * v->c2im[ip1] +
                u->c3.c2re[ip0] * v->c3re[ip1] - u->c3.c2im[ip0] * v->c3im[ip1];
   res->c2.im = -u->c1.c2re[ip0] * v->c1im[ip1] - u->c1.c2im[ip0] * v->c1re[ip1] +
                -u->c2.c2re[ip0] * v->c2im[ip1] - u->c2.c2im[ip0] * v->c2re[ip1] +
                -u->c3.c2re[ip0] * v->c3im[ip1] - u->c3.c2im[ip0] * v->c3re[ip1];
   res->c3.re = u->c1.c3re[ip0] * v->c1re[ip1] - u->c1.c3im[ip0] * v->c1im[ip1] +
                u->c2.c3re[ip0] * v->c2re[ip1] - u->c2.c3im[ip0] * v->c2im[ip1] +
                u->c3.c3re[ip0] * v->c3re[ip1] - u->c3.c3im[ip0] * v->c3im[ip1];
   res->c3.im = -u->c1.c3re[ip0] * v->c1im[ip1] - u->c1.c3im[ip0] * v->c1re[ip1] +
                -u->c2.c3re[ip0] * v->c2im[ip1] - u->c2.c3im[ip0] * v->c2re[ip1] +
                -u->c3.c3re[ip0] * v->c3im[ip1] - u->c3.c3im[ip0] * v->c3re[ip1];
}
#pragma omp end declare target


#pragma omp declare target
static void _fsu3matxsu3vec(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
{
   res->c1.re = u->c1.c1re[ip0] * v->c1re[ip1] - u->c1.c1im[ip0] * v->c1im[ip1] +
                u->c1.c2re[ip0] * v->c2re[ip1] - u->c1.c2im[ip0] * v->c2im[ip1] +
                u->c1.c3re[ip0] * v->c3re[ip1] - u->c1.c3im[ip0] * v->c3im[ip1];
   res->c1.im = u->c1.c1re[ip0] * v->c1im[ip1] + u->c1.c1im[ip0] * v->c1re[ip1] +
                u->c1.c2re[ip0] * v->c2im[ip1] + u->c1.c2im[ip0] * v->c2re[ip1] +
                u->c1.c3re[ip0] * v->c3im[ip1] + u->c1.c3im[ip0] * v->c3re[ip1];
   res->c2.re = u->c2.c1re[ip0] * v->c1re[ip1] - u->c2.c1im[ip0] * v->c1im[ip1] +
                u->c2.c2re[ip0] * v->c2re[ip1] - u->c2.c2im[ip0] * v->c2im[ip1] +
                u->c2.c3re[ip0] * v->c3re[ip1] - u->c2.c3im[ip0] * v->c3im[ip1];
   res->c2.im = u->c2.c1re[ip0] * v->c1im[ip1] + u->c2.c1im[ip0] * v->c1re[ip1] +
                u->c2.c2re[ip0] * v->c2im[ip1] + u->c2.c2im[ip0] * v->c2re[ip1] +
                u->c2.c3re[ip0] * v->c3im[ip1] + u->c2.c3im[ip0] * v->c3re[ip1];
   res->c3.re = u->c3.c1re[ip0] * v->c1re[ip1] - u->c3.c1im[ip0] * v->c1im[ip1] +
                u->c3.c2re[ip0] * v->c2re[ip1] - u->c3.c2im[ip0] * v->c2im[ip1] +
                u->c3.c3re[ip0] * v->c3re[ip1] - u->c3.c3im[ip0] * v->c3im[ip1];
   res->c3.im = u->c3.c1re[ip0] * v->c1im[ip1] + u->c3.c1im[ip0] * v->c1re[ip1] +
                u->c3.c2re[ip0] * v->c2im[ip1] + u->c3.c2im[ip0] * v->c2re[ip1] +
                u->c3.c3re[ip0] * v->c3im[ip1] + u->c3.c3im[ip0] * v->c3re[ip1];
}
#pragma omp end declare target


/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
#pragma omp declare target
void fsu3dagxsu3dag(
    const su3_mat_field *u, su3_dble *res, int ip0, int ip1)
{
   su3_vec_field psi;
   su3_vector_dble chi;

   psi=(*u).c1;
   _fsu3matconj_inverse_multiply(u,&psi,&chi,ip0,ip1);
   (*res).c11=chi.c1;
   (*res).c21=chi.c2;
   (*res).c31=chi.c3;

   psi=(*u).c2;
   _fsu3matconj_inverse_multiply(u,&psi,&chi,ip0,ip1);
   (*res).c12=chi.c1;
   (*res).c22=chi.c2;
   (*res).c32=chi.c3;

   psi=(*u).c3;
   _fsu3matconj_inverse_multiply(u,&psi,&chi,ip0,ip1);
   (*res).c13=chi.c1;
   (*res).c23=chi.c2;
   (*res).c33=chi.c3;
}
#pragma omp end declare target


#pragma omp declare target
void fsu3xsu3(
    const su3_mat_field *u, su3_dble *res, int ip0, int ip1)
{
   su3_vec_field psi;
   su3_vector_dble chi;

   psi.c1re=(*u).c1.c1re;
   psi.c1im=(*u).c1.c1im;
   psi.c2re=(*u).c2.c1re;
   psi.c2im=(*u).c2.c1im;
   psi.c3re=(*u).c3.c1re;
   psi.c3im=(*u).c3.c1im;
   _fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c11=chi.c1;
   (*res).c21=chi.c2;
   (*res).c31=chi.c3;

   psi.c1re=(*u).c1.c2re;
   psi.c1im=(*u).c1.c2im;
   psi.c2re=(*u).c2.c2re;
   psi.c2im=(*u).c2.c2im;
   psi.c3re=(*u).c3.c2re;
   psi.c3im=(*u).c3.c2im;
   _fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c12=chi.c1;
   (*res).c22=chi.c2;
   (*res).c32=chi.c3;

   psi.c1re=(*u).c1.c3re;
   psi.c1im=(*u).c1.c3im;
   psi.c2re=(*u).c2.c3re;
   psi.c2im=(*u).c2.c3im;
   psi.c3re=(*u).c3.c3re;
   psi.c3im=(*u).c3.c3im;
   _fsu3matxsu3vec(u,&psi,&chi,ip0,ip1);
   (*res).c13=chi.c1;
   (*res).c23=chi.c2;
   (*res).c33=chi.c3;
}
#pragma omp end declare target

#endif

