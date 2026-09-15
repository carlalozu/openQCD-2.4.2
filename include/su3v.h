
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

static inline void doublev_init(doublev *x, size_t volume)
{
    // Round up to nearest 8
    size_t padded_volume = (volume + 7) & ~7;
    size_t size = padded_volume * sizeof(double);
    x->volume = padded_volume;
    x->base = (double *)aligned_alloc(ALIGN, size);
    if (!x->base)
    {
        x->volume = 0;
        fprintf(stderr, "Error allocating complexv");
        abort();
    }
}

static inline void su3_vec_field_init(su3_vec_field *v, size_t volume)
{
    // Round up to nearest 8
    size_t padded_volume = (volume + 7) & ~7;
    v->volume = padded_volume;
    size_t size = 6 * padded_volume * sizeof(double);
    v->base = (double *)aligned_alloc(ALIGN, size);
    if (!v->base)
    {
        v->volume = 0;
        fprintf(stderr, "Error allocating su3_vec_field");
        abort();
    }

    v->c1re = v->base + 0 * padded_volume;
    v->c1im = v->base + 1 * padded_volume;
    v->c2re = v->base + 2 * padded_volume;
    v->c2im = v->base + 3 * padded_volume;
    v->c3re = v->base + 4 * padded_volume;
    v->c3im = v->base + 5 * padded_volume;
}

static inline void su3_mat_field_init(su3_mat_field *m_field, size_t volume)
{
    su3_vec_field_init(&m_field->c1, volume);
    su3_vec_field_init(&m_field->c2, volume);
    su3_vec_field_init(&m_field->c3, volume);
}

static inline void enter_double_field(doublev *d_field)
{
    double *base = d_field->base;
    size_t  vol  = (size_t)d_field->volume;

#pragma omp target enter data map(to : d_field[0:1], base[0:vol])
}

static inline void enter_su3_mat_field(su3_mat_field *m_field)
{
double *b1 = m_field->c1.base;  size_t n1 = 6 * (size_t)m_field->c1.volume;
double *b2 = m_field->c2.base;  size_t n2 = 6 * (size_t)m_field->c2.volume;
double *b3 = m_field->c3.base;  size_t n3 = 6 * (size_t)m_field->c3.volume;

#pragma omp target enter data map(to : b1[0:n1], b2[0:n2], b3[0:n3])

#pragma omp target
    {
        size_t volume = m_field->c1.volume;
        m_field->c1.c1re = b1 + 0 * volume;
        m_field->c1.c1im = b1 + 1 * volume;
        m_field->c1.c2re = b1 + 2 * volume;
        m_field->c1.c2im = b1 + 3 * volume;
        m_field->c1.c3re = b1 + 4 * volume;
        m_field->c1.c3im = b1 + 5 * volume;

        m_field->c2.c1re = b2 + 0 * volume;
        m_field->c2.c1im = b2 + 1 * volume;
        m_field->c2.c2re = b2 + 2 * volume;
        m_field->c2.c2im = b2 + 3 * volume;
        m_field->c2.c3re = b2 + 4 * volume;
        m_field->c2.c3im = b2 + 5 * volume;

        m_field->c3.c1re = b3 + 0 * volume;
        m_field->c3.c1im = b3 + 1 * volume;
        m_field->c3.c2re = b3 + 2 * volume;
        m_field->c3.c2im = b3 + 3 * volume;
        m_field->c3.c3re = b3 + 4 * volume;
        m_field->c3.c3im = b3 + 5 * volume;
    }
}

#endif
