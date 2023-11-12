CFLAGS = -Wall -Werror -std=gnu99 -g -Os

ZESO_PROGS = zesomunch helloworld echo keysym
LXLIB = -L. -lzeso -lX11 -lXext
ZESO_DEPS = libzeso.a zeso.h

all: $(ZESO_PROGS)

clean:
	$(RM) $(ZESO_PROGS) *.a *.o

libzeso.a: zeso.o yeso.o yeso.h yeso.c
	ar rcsDv $@ $^
zesomunch: zesomunch.o $(ZESO_DEPS)
	$(CC) $(CFLAGS) -o $@ $< $(LXLIB)
helloworld: helloworld.o $(ZESO_DEPS)
	$(CC) $(CFLAGS) -o $@ $< $(LXLIB)
echo: echo.o $(ZESO_DEPS)
	$(CC) $(CFLAGS) -o $@ $< $(LXLIB)
keysym: keysym.o $(ZESO_DEPS)
	$(CC) $(CFLAGS) -o $@ $< $(LXLIB)
