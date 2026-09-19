CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common
CLAW_PKG = dclaw
EXE = $(PWD)/xdclaw
SETRUN_FILE = setrun.py
OUTDIR = _output
FC = gfortran
FFLAGS ?= -O -gno-strict-dwarf -fbounds-check -fopenmp -std=legacy -ffpe-trap='invalid,overflow,zero'
DIGLIB = $(CLAW)/dclaw/src/2d/dig
EXCLUDE_SOURCES = $(DIGLIB)/src2.f90
SOURCES = src2.f90
include $(DIGLIB)/Makefile.dclaw
include $(CLAWMAKE)
