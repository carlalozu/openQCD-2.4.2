
/*******************************************************************************
*
* File uflds.h
*
* Copyright (C) 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef SU3V_H
#include "su3v.h"
#endif

#include "profiler.h"


/* BSTAP_C */
extern su3_dble *bstap(void);
extern void set_bstap(void);

/* PLAQ_SUM_C */
extern double local_plaq_dble(int n);
extern double plaq_sum_dble(int icom);
extern double plaq_wsum_dble(int icom);
extern double plaq_action_slices(double *asl);
extern prof_section lcl_plq_sm_s;
extern prof_section lcl_plq_smv_s;

/* PLAQ_SUMV_C */
extern double local_plaq_dblev(int n);
extern double plaq_sum_dblev(int icom);
extern double plaq_wsum_dblev(int icom);
extern double plaq_action_slicesv(double *asl);

/* SHIFT_C */
extern int shift_ud(int *s);

/* UDCOM_C */
extern void copy_bnd_ud(void);

/* UFLDS_C */
extern su3 *ufld(void);
extern su3_dble *udfld(void);
extern void random_ud(void);
extern void random_ud_reproducible(void);
extern void set_ud_phase(void);
extern void unset_ud_phase(void);
extern void renormalize_ud(void);
extern void assign_ud2u(void);

/* UFLDSV_C */
extern su3_mat_field *udfldv(void);
extern void random_udv(void);
extern void update_su3_mat_field(void);

/* UINIT_C */
extern void set_u2unity(int vol,int icom,su3 *u);
extern void set_ud2unity(int vol,int icom,su3_dble *ud);
extern void set_ud2unityv(int vol,int icom,su3_mat_field *udv);
extern void assign_ud2ud(int vol,int icom,su3_dble *ud,su3_dble *vd);

#endif
