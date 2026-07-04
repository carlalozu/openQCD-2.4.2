
/*******************************************************************************
*
* File su3fcts.h
*
* Copyright (C) 2010, 2011, 2016 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef SU3FCTS_H
#define SU3FCTS_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef SU3V_H
#include "su3v.h"
#endif

typedef struct
{
   double t,d;
   complex_dble p[3] ALIGNED16;
} ch_drv0_t;

typedef struct
{
   double t,d;
   complex_dble p[3] ALIGNED16;
   complex_dble pt[3] ALIGNED16;
   complex_dble pd[3] ALIGNED16;
} ch_drv1_t;

typedef struct
{
   double t,d;
   complex_dble p[3] ALIGNED16;
   complex_dble pt[3] ALIGNED16;
   complex_dble pd[3] ALIGNED16;
   complex_dble ptt[3] ALIGNED16;
   complex_dble ptd[3] ALIGNED16;
   complex_dble pdd[3] ALIGNED16;
} ch_drv2_t;

/* CHEXP_C */
extern void chexp_init(void);
extern void chexp_drv1(su3_alg_dble *X,ch_drv1_t *s);
extern void chexp_drv2(su3_alg_dble *X,ch_drv2_t *s);
#pragma omp declare target
extern void chexp_drv0(su3_alg_dble *X,ch_drv0_t *s);
extern void ch2mat(complex_dble *p,su3_alg_dble *X,su3_dble *u);
extern void expXsu3(double eps,su3_alg_dble *X,su3_dble *u);
#pragma omp end declare target

/* CM3X3_C */
extern void cm3x3_zero(int vol,su3_dble *u);
extern void cm3x3_unity(int vol,su3_dble *u);
extern void cm3x3_unityv(int vol,su3_mat_field *u,int i0);
extern void cm3x3_assign(int vol,su3_dble *u,su3_dble *v);
extern void cm3x3_swap(int vol,su3_dble *u,su3_dble *v);
extern void cm3x3_dagger(su3_dble *u,su3_dble *v);
extern void cm3x3_tr(su3_dble *u,su3_dble *v,complex_dble *tr);
#if defined (AVX)
extern void cm3x3_retr(su3_dble *u,su3_dble *v,double *tr);
#else
#pragma omp declare target
static inline void cm3x3_retr(su3_dble *u,su3_dble *v,double *tr)
{
   double r;

   r =(*u).c11.re*(*v).c11.re-(*u).c11.im*(*v).c11.im;
   r+=(*u).c12.re*(*v).c21.re-(*u).c12.im*(*v).c21.im;
   r+=(*u).c13.re*(*v).c31.re-(*u).c13.im*(*v).c31.im;

   r+=(*u).c21.re*(*v).c12.re-(*u).c21.im*(*v).c12.im;
   r+=(*u).c22.re*(*v).c22.re-(*u).c22.im*(*v).c22.im;
   r+=(*u).c23.re*(*v).c32.re-(*u).c23.im*(*v).c32.im;

   r+=(*u).c31.re*(*v).c13.re-(*u).c31.im*(*v).c13.im;
   r+=(*u).c32.re*(*v).c23.re-(*u).c32.im*(*v).c23.im;
   r+=(*u).c33.re*(*v).c33.re-(*u).c33.im*(*v).c33.im;

   (*tr)=r;
}
#pragma omp end declare target
#endif
extern void cm3x3_imtr(su3_dble *u,su3_dble *v,double *tr);
extern void cm3x3_add(su3_dble *u,su3_dble *v);
extern void cm3x3_mul_add(su3_dble *u,su3_dble *v,su3_dble *w);
extern void cm3x3_mulr(double *r,su3_dble *u,su3_dble *v);
extern void cm3x3_mulr_add(double *r,su3_dble *u,su3_dble *v);
extern void cm3x3_mulc(complex_dble *c,su3_dble *u,su3_dble *v);
extern void cm3x3_mulc_add(complex_dble *c,su3_dble *u,su3_dble *v);
extern void cm3x3_lc1(complex_dble *c,su3_dble *u,su3_dble *v);
extern void cm3x3_lc2(complex_dble *c,su3_dble *u,su3_dble *v);

/* RANDOM_SU3_C */
extern void random_su3(su3 *u);
extern void random_su3_dble(su3_dble *u);
extern void random_su3_mat_field(su3_mat_field *uv, int ix);

/* SU3REN_C */
extern void project_to_su3(su3 *u);
extern void project_to_su3_dble(su3_dble *u);

/* SU3PROD_C */
#if defined (AVX)
extern void su3xsu3(su3_dble *u,su3_dble *v,su3_dble *w);
extern void su3dagxsu3(su3_dble *u,su3_dble *v,su3_dble *w);
extern void su3xsu3dag(su3_dble *u,su3_dble *v,su3_dble *w);
extern void su3dagxsu3dag(su3_dble *u,su3_dble *v,su3_dble *w);
extern double prod2su3alg(su3_dble *u,su3_dble *v,su3_alg_dble *X);
#else
#pragma omp declare target
static inline void su3xsu3vec(su3_dble *u,su3_vector_dble *psi,
                       su3_vector_dble *chi)
{
   _su3_multiply(*chi,*u,*psi);
}
#pragma omp end declare target

#pragma omp declare target
static inline void su3dagxsu3vec(su3_dble *u,su3_vector_dble *psi,
                          su3_vector_dble *chi)
{
   _su3_inverse_multiply(*chi,*u,*psi);
}
#pragma omp end declare target

#pragma omp declare target
static inline void su3xsu3(su3_dble *u,su3_dble *v,su3_dble *w)
{
   su3_vector_dble psi,chi;

   psi.c1=(*v).c11;
   psi.c2=(*v).c21;
   psi.c3=(*v).c31;
   su3xsu3vec(u,&psi,&chi);
   (*w).c11=chi.c1;
   (*w).c21=chi.c2;
   (*w).c31=chi.c3;

   psi.c1=(*v).c12;
   psi.c2=(*v).c22;
   psi.c3=(*v).c32;
   su3xsu3vec(u,&psi,&chi);
   (*w).c12=chi.c1;
   (*w).c22=chi.c2;
   (*w).c32=chi.c3;

   psi.c1=(*v).c13;
   psi.c2=(*v).c23;
   psi.c3=(*v).c33;
   su3xsu3vec(u,&psi,&chi);
   (*w).c13=chi.c1;
   (*w).c23=chi.c2;
   (*w).c33=chi.c3;
}
#pragma omp end declare target

#pragma omp declare target
static inline void su3dagxsu3(su3_dble *u,su3_dble *v,su3_dble *w)
{
   su3_vector_dble psi,chi;

   psi.c1=(*v).c11;
   psi.c2=(*v).c21;
   psi.c3=(*v).c31;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c11=chi.c1;
   (*w).c21=chi.c2;
   (*w).c31=chi.c3;

   psi.c1=(*v).c12;
   psi.c2=(*v).c22;
   psi.c3=(*v).c32;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c12=chi.c1;
   (*w).c22=chi.c2;
   (*w).c32=chi.c3;

   psi.c1=(*v).c13;
   psi.c2=(*v).c23;
   psi.c3=(*v).c33;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c13=chi.c1;
   (*w).c23=chi.c2;
   (*w).c33=chi.c3;
}
#pragma omp end declare target

#pragma omp declare target
static inline void su3xsu3dag(su3_dble *u,su3_dble *v,su3_dble *w)
{
   su3_vector_dble psi,chi;

   psi.c1.re= (*v).c11.re;
   psi.c1.im=-(*v).c11.im;
   psi.c2.re= (*v).c12.re;
   psi.c2.im=-(*v).c12.im;
   psi.c3.re= (*v).c13.re;
   psi.c3.im=-(*v).c13.im;
   su3xsu3vec(u,&psi,&chi);
   (*w).c11=chi.c1;
   (*w).c21=chi.c2;
   (*w).c31=chi.c3;

   psi.c1.re= (*v).c21.re;
   psi.c1.im=-(*v).c21.im;
   psi.c2.re= (*v).c22.re;
   psi.c2.im=-(*v).c22.im;
   psi.c3.re= (*v).c23.re;
   psi.c3.im=-(*v).c23.im;
   su3xsu3vec(u,&psi,&chi);
   (*w).c12=chi.c1;
   (*w).c22=chi.c2;
   (*w).c32=chi.c3;

   psi.c1.re= (*v).c31.re;
   psi.c1.im=-(*v).c31.im;
   psi.c2.re= (*v).c32.re;
   psi.c2.im=-(*v).c32.im;
   psi.c3.re= (*v).c33.re;
   psi.c3.im=-(*v).c33.im;
   su3xsu3vec(u,&psi,&chi);
   (*w).c13=chi.c1;
   (*w).c23=chi.c2;
   (*w).c33=chi.c3;
}
#pragma omp end declare target

#pragma omp declare target
static inline void su3dagxsu3dag(su3_dble *u,su3_dble *v,su3_dble *w)
{
   su3_vector_dble psi,chi;

   psi.c1.re= (*v).c11.re;
   psi.c1.im=-(*v).c11.im;
   psi.c2.re= (*v).c12.re;
   psi.c2.im=-(*v).c12.im;
   psi.c3.re= (*v).c13.re;
   psi.c3.im=-(*v).c13.im;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c11=chi.c1;
   (*w).c21=chi.c2;
   (*w).c31=chi.c3;

   psi.c1.re= (*v).c21.re;
   psi.c1.im=-(*v).c21.im;
   psi.c2.re= (*v).c22.re;
   psi.c2.im=-(*v).c22.im;
   psi.c3.re= (*v).c23.re;
   psi.c3.im=-(*v).c23.im;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c12=chi.c1;
   (*w).c22=chi.c2;
   (*w).c32=chi.c3;

   psi.c1.re= (*v).c31.re;
   psi.c1.im=-(*v).c31.im;
   psi.c2.re= (*v).c32.re;
   psi.c2.im=-(*v).c32.im;
   psi.c3.re= (*v).c33.re;
   psi.c3.im=-(*v).c33.im;
   su3dagxsu3vec(u,&psi,&chi);
   (*w).c13=chi.c1;
   (*w).c23=chi.c2;
   (*w).c33=chi.c3;
}
#pragma omp end declare target

#pragma omp declare target
static inline double prod2su3alg(su3_dble *u,su3_dble *v,su3_alg_dble *X)
{
   double tr;
   su3_vector_dble psi,chi1,chi2,chi3;

   psi.c1=(*v).c11;
   psi.c2=(*v).c21;
   psi.c3=(*v).c31;
   su3xsu3vec(u,&psi,&chi1);
   tr=chi1.c1.re;
   (*X).c1 = chi1.c1.im;
   (*X).c2 = chi1.c1.im;
   (*X).c3 =-chi1.c2.re;
   (*X).c4 = chi1.c2.im;
   (*X).c5 =-chi1.c3.re;
   (*X).c6 = chi1.c3.im;

   psi.c1=(*v).c12;
   psi.c2=(*v).c22;
   psi.c3=(*v).c32;
   su3xsu3vec(u,&psi,&chi2);
   tr+=chi2.c2.re;
   (*X).c3+= chi2.c1.re;
   (*X).c4+= chi2.c1.im;
   (*X).c1-= chi2.c2.im;
   (*X).c7 =-chi2.c3.re;
   (*X).c8 = chi2.c3.im;

   psi.c1=(*v).c13;
   psi.c2=(*v).c23;
   psi.c3=(*v).c33;
   su3xsu3vec(u,&psi,&chi3);
   tr+=chi3.c3.re;
   (*X).c5+= chi3.c1.re;
   (*X).c6+= chi3.c1.im;
   (*X).c7+= chi3.c2.re;
   (*X).c8+= chi3.c2.im;
   (*X).c2-= chi3.c3.im;

   (*X).c1*=(1.0/3.0);
   (*X).c2*=(1.0/3.0);
   (*X).c3*=0.5;
   (*X).c4*=0.5;
   (*X).c5*=0.5;
   (*X).c6*=0.5;
   (*X).c7*=0.5;
   (*X).c8*=0.5;

   return tr;
}
#pragma omp end declare target

#endif
extern void su3xu3alg(su3_dble *u,u3_alg_dble *X,su3_dble *v);
extern void su3dagxu3alg(su3_dble *u,u3_alg_dble *X,su3_dble *v);
extern void u3algxsu3(u3_alg_dble *X,su3_dble *u,su3_dble *v);
extern void u3algxsu3dag(u3_alg_dble *X,su3_dble *u,su3_dble *v);
extern void prod2u3alg(su3_dble *u,su3_dble *v,u3_alg_dble *X);
extern void rotate_su3alg(su3_dble *u,su3_alg_dble *X);
extern void lieprod_u3alg(u3_alg_dble *X,u3_alg_dble *Y,u3_alg_dble *Z);

#pragma omp declare target
static inline void _fsu3matconj_inverse_multiply(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
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
static inline void _fsu3matxsu3vec(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
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
static inline void fsu3dagxsu3dag(
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
static inline void fsu3xsu3(
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
