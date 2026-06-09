################################################################################
#
# Makefile for tests/gpu/uflds
#
# Unit tests for the GPU-offloaded plaquette-sum routines in modules/uflds/.
#
# "make"       compiles and links all check programs
# "make clean" removes all generated files
#
################################################################################

all: rmxeq mkdep mkxeq
.PHONY: all


# main programs and modules to be compiled

MAIN = check4

DEVFCTS = dutils gtrans

FLAGS = flags lat_parms dfl_parms

LATTICE = bcnds geogen geometry uidx

RANDOM = ranlux ranlux_common ranlxs ranlxd gauss

SU3FCTS = su3prod su3ren cm3x3 random_su3

UFLDS = bstap plaq_sum shift udcom uflds uinit

UTILS = endian error mutils qsum utils wspace

MODULES = $(DEVFCTS) $(FLAGS) $(LATTICE) $(RANDOM) $(UFLDS) \
          $(SU3FCTS) $(UTILS)


# search path for modules

MDIR = ../../../modules

VPATH = .:$(MDIR)/devfcts:$(MDIR)/flags:$(MDIR)/lattice:$(MDIR)/random:\
          $(MDIR)/uflds:$(MDIR)/su3fcts:$(MDIR)/utils


# additional include directories

INCPATH = $(MPI_INCLUDE) ../../../include ../..


# additional libraries

LIBS = m

LIBPATH = $(MPI_HOME)/lib


# compiler and flags (GPU offload via OpenMP target)

CFLAGS = -std=c11 -pedantic -fstrict-aliasing -fopenmp -fopenmp-cuda-mode \
         -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_90 -Wall -Wno-long-long \
         -Wstrict-prototypes -O -DFMA3 -DPM

LFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -lmpi_gtl_cuda


############################## do not change ###################################

SHELL=/bin/bash
CC=mpicc
CLINKER=$(CC)

PGMS= $(MAIN) $(MODULES)

-include $(addsuffix .d,$(PGMS))


# rule to make dependencies

$(addsuffix .d,$(PGMS)): %.d: %.c Makefile
	@ $(GCC) -ansi $< -MM $(addprefix -I,$(INCPATH)) -o $@


# rule to compile source programs

$(addsuffix .o,$(PGMS)): %.o: %.c Makefile
	$(CC) $< -c $(CFLAGS) $(addprefix -I,$(INCPATH))


# rule to link object files

$(MAIN): %: %.o $(addsuffix .o,$(MODULES)) Makefile
	$(CLINKER) $< $(addsuffix .o,$(MODULES)) $(LFLAGS) \
        $(addprefix -L,$(LIBPATH)) $(addprefix -l,$(LIBS)) -o $@


# produce executables

mkxeq: $(MAIN)


# remove old executables

rmxeq:
	@ -rm -f $(MAIN); \
        echo "delete old executables"


# make dependencies

mkdep: $(addsuffix .d,$(PGMS))
	@ echo "generate tables of dependencies"


# clean directory

clean:
	@ -rm -rf *.d *.o *.log $(MAIN)
.PHONY: clean

################################################################################
