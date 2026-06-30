#pragma once
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "mpi.h"

#ifndef PROFILER_H
#define PROFILER_H

static inline double prof_now(void) {
  return MPI_Wtime();
}

typedef struct {
  const char *name;
  double total;
  int64_t count;
  double t0;
  int threads;
  int level;
} prof_section;

static inline void prof_begin(prof_section *s) { s->t0 = prof_now(); }

static inline void prof_end(prof_section *s) {
  double t1 = prof_now();
  s->total += (t1 - s->t0);
  s->count += 1;
}

static inline void prof_reset(prof_section *s) {
  s->total = 0.0;
  s->count = 0;
}

static inline void prof_report(const prof_section *s) {
  double avg = (s->count > 0) ? (s->total / (double)s->count) : 0.0;
  int indent = s->level * 2;
  printf("%*s%-*s | Total: %12.6f s | Avg: %12.6f s | Reps: %lld\n",
         indent, "", 24 - indent, s->name, s->total, avg, (long long)s->count);
}

#endif
