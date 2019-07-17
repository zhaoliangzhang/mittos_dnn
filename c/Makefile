main: test.c linear.o
	@gcc -O3 -g -o main test.c linear.o -lpthread

linear.o: linear.c
	@gcc -O3 -c linear.c

clean:
	@rm main linear.o