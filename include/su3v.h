
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
void complex_field_map_pointers(complexv *v)
{
    size_t volume = v->volume;
    v->re = v->base + 0*volume;
    v->im = v->base + 1*volume;
}
#pragma omp end declare target

#pragma omp declare target
void su3_vec_field_map_pointers(su3_vec_field *v)
{
    size_t volume = v->volume;
    v->c1re = v->base + 0*volume;
    v->c1im = v->base + 1*volume;
    v->c2re = v->base + 2*volume;
    v->c2im = v->base + 3*volume;
    v->c3re = v->base + 4*volume;
    v->c3im = v->base + 5*volume;
}
#pragma omp end declare target

#pragma omp declare target
void su3_mat_field_map_pointers(su3_mat_field *m)
{
    size_t volume = m->c1.volume;
    m->c1.c1re = m->c1.base + 0*volume;
    m->c1.c1im = m->c1.base + 1*volume;
    m->c1.c2re = m->c1.base + 2*volume;
    m->c1.c2im = m->c1.base + 3*volume;
    m->c1.c3re = m->c1.base + 4*volume;
    m->c1.c3im = m->c1.base + 5*volume;

    m->c2.c1re = m->c2.base + 0*volume;
    m->c2.c1im = m->c2.base + 1*volume;
    m->c2.c2re = m->c2.base + 2*volume;
    m->c2.c2im = m->c2.base + 3*volume;
    m->c2.c3re = m->c2.base + 4*volume;
    m->c2.c3im = m->c2.base + 5*volume;

    m->c3.c1re = m->c3.base + 0*volume;
    m->c3.c1im = m->c3.base + 1*volume;
    m->c3.c2re = m->c3.base + 2*volume;
    m->c3.c2im = m->c3.base + 3*volume;
    m->c3.c3re = m->c3.base + 4*volume;
    m->c3.c3im = m->c3.base + 5*volume;
}
#pragma omp end declare target

void enter_complex_field(complexv* c_field){
    #pragma omp target enter data \
    map(to : c_field[0], c_field->base[0:2*c_field->volume])

    #pragma omp target
    complex_field_map_pointers(c_field);

}

void enter_double_field(doublev* d_field){
    #pragma omp target enter data \
    map(to : d_field[0], d_field->base[0:d_field->volume])
}

void enter_su3_vec_field(su3_vec_field* v_field){
    #pragma omp target enter data \
    map(to : v_field[0], v_field->base[0:6*v_field->volume])

    #pragma omp target
    su3_vec_field_map_pointers(v_field);

}

void enter_su3_mat_field(su3_mat_field* m_field){
    #pragma omp target enter data map(to: m_field[0]) \
    map(to: m_field->c1.base[0 : 6*m_field->c1.volume]) \
    map(to: m_field->c2.base[0 : 6*m_field->c2.volume]) \
    map(to: m_field->c3.base[0 : 6*m_field->c3.volume])

    #pragma omp target
    su3_mat_field_map_pointers(m_field);
}

void enter_complex_field_array(complexv* c_field, int n_blocks) {
    #pragma omp target enter data map(to: c_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 2 * c_field[i].volume;
        #pragma omp target enter data map(to: c_field[i].base[0:total_size])

        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) { 
            // fake loop to trigger update done by each thread
            complex_field_map_pointers(&c_field[i]);
        }
    }
}

void update_host_complex_field_array(complexv* c_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 2 * c_field[i].volume;
        
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(c_field[i].base[0:total_size])
    }
}

void enter_double_field_array(doublev* d_field, int n_blocks) {
    #pragma omp target enter data map(to: d_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = d_field[i].volume;
        #pragma omp target enter data map(to: d_field[i].base[0:total_size])
    }
}


void update_host_double_field_array(doublev* d_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(d_field[i].base[0:d_field[i].volume])
    }
}

void enter_su3_vec_field_array(su3_vec_field* v_field, int n_blocks) {
    // map the array of structs itself
    #pragma omp target enter data map(to: v_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * v_field[i].volume;
        
        // map the data buffer for each struct
        // this also handles the "attachment" so v_field[i].base on the 
        // device points to the device-allocated buffer
        #pragma omp target enter data map(to: v_field[i].base[0:total_size])

        // fix the internal pointers (c1re, c1im, etc.) on the device
        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) { 
            // fake loop to trigger update done by each thread
            su3_vec_field_map_pointers(&v_field[i]);
        }
    }
}

void update_host_su3_vec_field_array(su3_vec_field* v_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * v_field[i].volume;
        
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(v_field[i].base[0:total_size])
    }
}

void enter_su3_mat_field_array(su3_mat_field* m_field, int n_blocks) {
    // map the array of structs itself
    #pragma omp target enter data map(to: m_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * m_field[i].c1.volume;

        // map the data buffer for each struct
        #pragma omp target enter data map(to: m_field[i].c1.base[0:total_size])
        #pragma omp target enter data map(to: m_field[i].c2.base[0:total_size])
        #pragma omp target enter data map(to: m_field[i].c3.base[0:total_size])

        // fix the internal pointers on the device
        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) {
            su3_mat_field_map_pointers(&m_field[i]);
        }
    }
}

void update_host_su3_mat_field_array(su3_mat_field* m_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * m_field[i].c1.volume;
        
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(m_field[i].c1.base[0:total_size])
        #pragma omp target update from(m_field[i].c2.base[0:total_size])
        #pragma omp target update from(m_field[i].c3.base[0:total_size])
    }
}

#endif
