#pragma once
#include <stdio.h>
#include <stdint.h>
#include <time.h>

static inline double prof_now(void) {
  return MPI_Wtime();
}

typedef struct {
  const char *name;
  double total;
  int64_t count;
  double t0;
  int threads;
} prof_section;

static inline void prof_begin(prof_section *s) { s->t0 = prof_now(); }
static inline void prof_end(prof_section *s) {
  double t1 = prof_now();
  s->total += (t1 - s->t0);
  s->count += 1;
}

static inline void prof_report(const prof_section *s) {
  double avg = (s->count > 0) ? (s->total / (double)s->count) : 0.0;
  printf("%-24s | Total: %12.6f s | Avg: %12.6f s | Reps: %lld\n", 
        s->name, s->total, avg, (long long)s->count);
}
