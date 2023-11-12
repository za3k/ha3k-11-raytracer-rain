/** Really dumb windowing library, providing just a shared-memory
 * framebuffer you can flush onto the screen and some events, with
 * different backends.  See
 * README.md for more details.
 */

#ifndef YESO_H_INCLUDED
#define YESO_H_INCLUDED
#include <stdint.h>

typedef struct _ywin *ywin;

/** Call yw_open to connect to the X11 display, open a window, and
 * set up a framebuffer for it.  This function takes care of
 * allocating and initializing the data structure.  The options string
 * is kind of a gateway for future options.  Passing an empty string
 * is safe.
 */
ywin yw_open(const char *name, int width, int height, const char *options);

/** Call yw_close to disconnect if the process is going to continue.
 */
void yw_close(ywin w);

/* ypix is a 32-bit little-endian RGB TrueColor pixel: the
 * least-significant byte is blue, the next is green, the next is red,
 * and the most significant byte is ignored.  Maybe you could compile
 * some Yeso apps for a platform with smaller pixels by changing this
 * typedef.
 */
typedef uint32_t ypix;

/** ypic — a TrueColor in-memory image — either a framebuffer, or not.
 * (It may be just a plain memory buffer or part of a larger ypic.)
 */
typedef struct { ypix *p; int w, stride, h; } ypic;

/** Call yw_frame to get a framebuffer ypic to paint in.
 * The width and height may be different from what you requested in
 * `yw_open` because the user may have resized the window, or because
 * you’re running on the raw hardware framebuffer and therefore you
 * got the whole screen.)
 */
ypic yw_frame(ywin w);

/** Call yw_flip to make visible on the screen the image you
 * have drawn in the framebuffer.
 */
void yw_flip(ywin w);


/** yp_pix computes the address of a pixel in a ypic.  No bounds check.
 */
static inline ypix *
yp_pix(ypic c, int x, int y)
{
  return &c.p[x + y*c.stride];
}

/** Create an in-memory ypic.
 */
ypic yp_new(int w, int h);

/** Represent a subrectangle of a ypic as a ypic.
 * Unlike yp_pix, this does do bounds-checking and clipping.  This
 * function is the reason `stride` and `w` are separate in ypic:
 * a 10×10 yp_sub of a 1024×1024 canvas or framebuffer still has
 * a stride of 1024.
 */
ypic yp_sub(ypic base, int x, int y, int w, int h);

/** Copy (possibly overlapping) image rectangles.
 */
void yp_copy(ypic dest, ypic src);

/** Fill an image rectangle.
 */

static inline void
yp_fill(ypic pic, ypix color)
{
  for (int i = 0; i < pic.h; i++) {
    ypix *p = yp_pix(pic, 0, i);
    for (int j = 0; j < pic.w; j++) p[j] = color;
  }
}


/** The opaque `yw_event` type returned by `yw_get_event` is
 * polymorphic; it can contain one of `yw_mouse_event`,
 * `yw_key_event`, `yw_die_event`, or `yw_raw_event`.  Call the associated
 * `yw_as_` function for each event type you want to handle.  It
 * will return NULL if the event is of the wrong type.
 *
 * Currently these event types are mutually exclusive, but they offer
 * the possibility of adding event subtypes at some point — either to
 * add more information, such as timestamps or keycodes, or to provide
 * subsets of existing event types, such as a hypothetical
 * `yw_as_keydown_event`.
 */
typedef struct yw_event yw_event;

/** Call `yw_get_event` to get the next input event, or NULL if there
 * is none yet.  The pointer this returns is to the `yw_event` allocated
 * within the ywin struct, so it gets overwritten the next time you
 * call this function.  This function does not block!
 */
yw_event *yw_get_event(ywin w);

/** Call yw_wait to block the program until an input event arrives, or
 * if a nonzero timeout is specified, for that number of
 * microseconds if no event appears.
 */
void yw_wait(ywin w, long timeout_usecs);

typedef struct {
  int32_t x, y;         /* Mouse coordinates relative to upper left */
  uint8_t buttons;      /* Bitmask of buttons that are down */
} yw_mouse_event;

yw_mouse_event *yw_as_mouse_event(yw_event *);

typedef struct {
  char *s;               /* pointer to UTF-8 bytes generated */
  uint64_t keysym;       /* X11 standard (?) keysym (+ bit 32 if ctrl held) */
  uint8_t down;          /* 0 if this is a keyup, 1 if keydown */
  uint8_t len;           /* number of bytes in string s */
} yw_key_event;

yw_key_event *yw_as_key_event(yw_event *);

/* useful keysyms */
enum { yk_left = 0xff51, yk_up, yk_right, yk_down,
       yk_pgup, yk_pgdn, yk_end, yk_home,
       yk_shift_l = 0xffe1, yk_shift_r, yk_control_l, yk_control_r,
       yk_caps_lock, yk_shift_lock, yk_meta_l, yk_meta_r,
       yk_backspace = 0xff08, yk_return = 0xff0d };

typedef void *yw_raw_event; /* secretly an XEvent pointer, for the X11 backend */

yw_raw_event *yw_as_raw_event(yw_event *);

typedef int yw_die_event;  /* False unless the event is a die event */

yw_die_event yw_as_die_event(yw_event *);

/** Call yw_fds to get the Unix file descriptors of the X11
 * connection or other input stream used by a ywin
 * so that you can use it in select() or epoll or whatever.
 * Keep in mind that there may already be events pending that have
 * been read from the file descriptor the last time you called
 * yw_wait or yw_get_event.
 */

void yw_fds(ywin w, void (*callback)(void *, int fd), void *);

/** If you set yeso_error_handler to point to a function of yours, it
 * will be invoked with a message when there’s an error; the default
 * function prints a message to stderr and calls abort().
 */
typedef void (*yeso_error_handler_t)(const char *msg);
extern yeso_error_handler_t yeso_error_handler;

#endif  /* YESO_H */
