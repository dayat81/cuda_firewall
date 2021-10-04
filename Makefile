CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)

rte_bv:
	nvcc -forward-unknown-to-host-compiler -Werror all-warnings $(CFLAGS) -O3 -c rte_bv.c

rte_table_bv:
	nvcc -forward-unknown-to-host-compiler -Werror all-warnings $(CFLAGS) -O3 -c rte_table_bv.cu

unittest_rte_bv: rte_bv
	nvcc -forward-unknown-to-host-compiler -Werror all-warnings  $(CFLAGS) -o unittest_rte_bv unittest_rte_bv.c rte_bv.o $(CLIBS)

firewall: rte_bv rte_table_bv
	nvcc -forward-unknown-to-host-compiler -mssse3 -Werror  cross-execution-space-call,default-stream-launch,ext-lambda-captures-this,reorder  $(CFLAGS) -o firewall firewall.c rte_table_bv.o rte_bv.o parser.c $(CLIBS) 

clean:
	rm *.o firewall

all: firewall