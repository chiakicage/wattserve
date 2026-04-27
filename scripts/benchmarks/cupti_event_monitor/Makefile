CUDA_HOME ?= /usr/local/cuda-12.4
CXX ?= g++

CUDART_INCLUDE := $(CUDA_HOME)/include
CUPTI_INCLUDE := $(CUDA_HOME)/extras/CUPTI/include
CUPTI_LIB := $(CUDA_HOME)/extras/CUPTI/lib64

CXXFLAGS ?= -O2 -std=c++17 -fPIC
LDFLAGS ?= -shared
LDLIBS := -L$(CUDA_HOME)/lib64 -L$(CUPTI_LIB) -lcupti -lcuda -lpthread

all: libcupti_active_event_monitor.so

libcupti_active_event_monitor.so: cupti_active_event_monitor.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDART_INCLUDE) -I$(CUPTI_INCLUDE) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f libcupti_active_event_monitor.so

.PHONY: all clean
