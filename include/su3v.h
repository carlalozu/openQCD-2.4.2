
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
    complex_dble *base;
    complex_dble *c1;
    complex_dble *c2;
    complex_dble *c3;
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
    size_t size = 3 * padded_volume * sizeof(complex_dble);
    v->base = (complex_dble *)aligned_alloc(ALIGN, size);
    if (!v->base)
    {
        v->volume = 0;
        fprintf(stderr, "Error allocating su3_vec_field");
        abort();
    }

    v->c1 = v->base + 0 * padded_volume;
    v->c2 = v->base + 1 * padded_volume;
    v->c3 = v->base + 2 * padded_volume;
}

static inline void su3_mat_field_init(su3_mat_field *m_field, size_t volume)
{
    su3_vec_field_init(&m_field->c1, volume);
    su3_vec_field_init(&m_field->c2, volume);
    su3_vec_field_init(&m_field->c3, volume);
}

static inline void enter_double_field(doublev *d_field)
{
#pragma omp target enter data \
map(to : d_field[0], d_field->base[0 : d_field->volume])
}

static inline void enter_su3_mat_field(su3_mat_field *m_field)
{
#pragma omp target enter data map(to : m_field->c1.base[0 : 3 * m_field->c1.volume], \
                                       m_field->c2.base[0 : 3 * m_field->c2.volume], \
                                       m_field->c3.base[0 : 3 * m_field->c3.volume])

#pragma omp target
    {
        size_t volume = m_field->c1.volume;
        m_field->c1.c1 = m_field->c1.base + 0 * volume;
        m_field->c1.c2 = m_field->c1.base + 1 * volume;
        m_field->c1.c3 = m_field->c1.base + 2 * volume;

        m_field->c2.c1 = m_field->c2.base + 0 * volume;
        m_field->c2.c2 = m_field->c2.base + 1 * volume;
        m_field->c2.c3 = m_field->c2.base + 2 * volume;

        m_field->c3.c1 = m_field->c3.base + 0 * volume;
        m_field->c3.c2 = m_field->c3.base + 1 * volume;
        m_field->c3.c3 = m_field->c3.base + 2 * volume;
    }
}

#endif
