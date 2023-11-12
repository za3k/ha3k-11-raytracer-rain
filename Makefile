CFLAGS=-Wall

LXLIB=-Iyeso -L/home/zachary/ha3k-11-raytracer-rain/yeso -lyeso-xlib -lX11 -lXext

raytrace-cuda: raytrace.cu yeso/yeso.h yeso/libyeso-xlib.a
	NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-12' nvcc -O5 $(LXLIB) -o $@ $<
clean:
	rm -f *.o yeso/*.o *.a yeso/*.a raytrace-cuda

yeso/libyeso-xlib.a: yeso/yeso.o
	ar rcsDv $@ $^
