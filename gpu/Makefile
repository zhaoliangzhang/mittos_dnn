NCC = nvcc

LDFLAGES := --cudart shared

EXE = main

OBJS = test.o

%.o: %.cu
	@$(NCC) -c $< -o $@ $(CFLAGS)

$(EXE): $(OBJS)
	@$(NCC) test.cu -o $@ $(LDFLAGES)

clean:
	@rm -f main *.o