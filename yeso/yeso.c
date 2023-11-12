// Implementation of Yeso functions for ypics, which are
// backend-independent.
#include <stdlib.h>
#include <string.h>

#include "yeso.h"

ypic
yp_sub(ypic base, int x, int y, int w, int h)
{
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (w > base.w - x) w = base.w - x;
  if (h > base.h - y) h = base.h - y;
  if (w < 0) w = 0;
  if (h < 0) h = 0;
  ypic result = { base.p + y*base.stride + x, w, base.stride, h };
  return result;
}

void
yp_copy(ypic dest, ypic src)
{
  int w = src.w, h = src.h;
  if (w > dest.w) w = dest.w;
  if (h > dest.h) h = dest.h;

  if ((size_t)((uintptr_t)src.p - (uintptr_t)dest.p)
      <= src.stride * (src.h + 1)) {
    // Possible vertical overlap requires copying upwards
    for (int y = h-1; y >= 0; y--) {
      memmove(&dest.p[y * dest.stride], &src.p[y * src.stride], w * sizeof(ypix));
    }

  } else {
    for (int y = 0; y < h; y++) {
      memmove(&dest.p[y * dest.stride], &src.p[y * src.stride], w * sizeof(ypix));
    }
  }
}

ypic
yp_new(int w, int h)
{
  ypic result = { malloc(w*h*sizeof(ypix)), w, w, h };
  if (!result.p) yeso_error_handler("canvas malloc");
  return result;
}

/* Get an X11 window on the screen and displaying stuff in the
 * simplest way possible.  Unfortunately, this is X11, so it isn’t
 * very simple; remember xiafpos.
 *
 * This program takes a different approach from xiafpos (q.v.) in that
 * it uses the X Shared Memory extension in order to provide, as far
 * as possible, just a dumb framebuffer interface.
 *
 * See README.md for more details.
 *
 * Doesn’t handle window resizes or copy-and-paste or drag-and-drop
 * yet.
 *
 * Since I didn’t have the X11 manual handy when I originally wrote
 * this, much of the code closely followed the z3d.l library from
 * picoLisp, which bears the following copyright notice:
 *
 * PicoLisp Copyright (c) Software Lab. Alexander Burger
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * (End of PicoLisp copyright notice.)
 *
 * This program is copyright (c) 2018–2019 Kragen Javier Sitaker, who is me.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/time.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>

#include "yeso.h"

static void
default_error_handler(const char *msg)
{
  fprintf(stderr, "yeso: %s\n", msg);
  abort();
}

yeso_error_handler_t yeso_error_handler = default_error_handler;

/* The type yw_event, declared but not defined in yeso.h. */
struct yw_event {
  enum { yw_raw = 0, yw_key, yw_mouse, yw_die } type;
  union {
    yw_key_event key;
    yw_mouse_event mouse;
    yw_raw_event raw;
  } body;
  ywin w;                       /* pointer for yw_as_die_event */
};

/* The destination type of the opaque pointer type ywin, declared but
 * not defined in yeso.h */
struct _ywin {
  Display *display;
  int screen;
  Colormap colormap;
  int depth, pixsize, width, height;
  GC gc;
  Window window;
  XImage *image;
  XShmSegmentInfo shm_segment_info;
  Atom WM_PROTOCOLS, WM_DELETE_WINDOW;
  XPixmapFormatValues *formats;  // XXX not even using this!
  int fbsize;                    // saved for debugging

  /* The most recently received event. */
  XEvent xevent;
  yw_event event;
  uint8_t has_event;  // True if there’s an unprocessed event in event
  uint8_t valid;      // True if client called yw_frame after yw_flip
  uint8_t handles_die;  // True if client tries to handle die events
  char keybuf[16];
};

ywin
yw_open(const char *name, int width, int height, const char *options)
{
  ywin w = malloc(sizeof(*w));
  char *err = "yw_open malloc failed";
  if (!w) goto fail;
  memset(w, 0, sizeof(*w));

  err = "XOpenDisplay failed";
  if (!(w->display = XOpenDisplay(NULL))) goto fail;
  w->screen = DefaultScreen(w->display);
  w->colormap = DefaultColormap(w->display, w->screen);
  int n_pixel_formats = 0;
  w->formats = XListPixmapFormats(w->display, &n_pixel_formats);
  err = "no pixel formats";
  if (n_pixel_formats == 0) goto fail;

  int best_index = 0;
  for (int i = 1; i < n_pixel_formats; i++) {
    if (w->formats[i].depth > w->formats[best_index].depth) best_index = i;
  }

  /* XXX the depth must match the depth of the drawable, i.e. the window */
  /* XCreateWindow allows you to specify the window depth, but also
     requires you to specify the class (InputOutput), the Visual, the
     valuemask (CWcrap; xfe uses CWBackPixel | CWBorderPixel |
     CWBitGravity | CWEventMask | CWColormap), and the
     XSetWindowAttributes (which includes .background_pixel,
     .border_pixel, .bit_gravity, .event_mask, and .colormap
     (XDefaultColormap(display, screen)), but can otherwise be
     zero).  */
  w->depth = 24;//formats[best_index].depth;
  w->pixsize = 4;//(formats[best_index].bits_per_pixel + 7) / 8;

  w->width = width;
  w->height = height;
  w->window = XCreateSimpleWindow(w->display, RootWindow(w->display, w->screen), 0, 0,
                                  width, height,
                                  0, /* border_width */
                                  BlackPixel(w->display, w->screen),
                                  WhitePixel(w->display, w->screen));
  Atom _NET_WM_NAME = XInternAtom(w->display, "_NET_WM_NAME", True),
    UTF8_STRING = XInternAtom(w->display, "UTF8_STRING", True);
  XChangeProperty(w->display, w->window, _NET_WM_NAME,
                  UTF8_STRING, 8, PropModeReplace,
                  (unsigned char*)name, strlen(name));

  w->WM_DELETE_WINDOW = XInternAtom(w->display, "WM_DELETE_WINDOW", True);
  w->WM_PROTOCOLS = XInternAtom(w->display, "WM_PROTOCOLS", True);

  XMapWindow(w->display, w->window);
  /* Apparently StructureNotifyMask is what you use to get MapNotify
   * events? */
  XSelectInput(w->display, w->window, KeyPressMask
               | KeyReleaseMask
               | ButtonPressMask
               | ButtonReleaseMask
               | PointerMotionMask
               | StructureNotifyMask
  );
  w->gc = XCreateGC(w->display, RootWindow(w->display, w->screen), 0, NULL);

  err = "no XShm";
  if (!XShmQueryExtension(w->display)) goto fail;

  /* Now wait for the window to open so that we can draw on it safely */
  w->xevent.type = 0;
  while (w->xevent.type != MapNotify) {
    yw_wait(w, 0);
    yw_get_event(w);
  }

  /* We wait to allocate the shared-memory framebuffer until the
   * client requests it with `yw_frame`, instead of doing it
   * here (as I did originally).  This way, it always has the most
   * up-to-date geometry.
   */

  return w;

 fail:
  yw_close(w);
  yeso_error_handler(err);
  return NULL;
}

/* Factored out because it gets called on close and on resize */
static void
destroy_framebuffer(ywin w)
{
  if (w->shm_segment_info.shmaddr && w->shm_segment_info.shmaddr != (char*)-1) { /* XXX use intptr_t */
    shmdt(w->shm_segment_info.shmaddr);
    w->shm_segment_info.shmaddr = 0;
  }

  if (w->image) XDestroyImage(w->image);
  w->image = 0;
}

void
yw_close(ywin w)
{
  if (!w) return;

  destroy_framebuffer(w);

  free(w->formats);
  if (w->gc) XFreeGC(w->display, w->gc);
  if (w->display) {
    XCloseDisplay(w->display);
    w->display = 0;
  }

  free(w);
}

static int
setup_framebuffer(ywin w, int width, int height)
{
  // If we have an existing framebuffer (presumably before a window
  // resize event) we need to clean it up first.
  if (w->image) destroy_framebuffer(w);

  // XXX are we guaranteed that the Window was created with the default visual?
  w->image = XShmCreateImage(w->display, DefaultVisual(w->display, w->screen),
                             w->depth, ZPixmap, NULL, &w->shm_segment_info,
                             width, height);
  char *err = "XShmCreateImage";
  if (!w->image) goto fail;

  w->fbsize = width * height * w->pixsize;
  w->shm_segment_info.shmid = shmget(IPC_PRIVATE, w->fbsize, IPC_CREAT | 0600);
  err = "shmget";
  if (w->shm_segment_info.shmid < 0) goto fail;
  w->shm_segment_info.shmaddr = shmat(w->shm_segment_info.shmid, 0, 0);
  err = "shmat";
  if (w->shm_segment_info.shmaddr == (char*)-1) goto fail; /* XXX use intptr_t */
  w->image->data = w->shm_segment_info.shmaddr;
  err = "XShmAttach";
  if (!XShmAttach(w->display, &w->shm_segment_info)) goto fail;

  w->width = width;
  w->height = height;

  return 1;
 fail:
  yeso_error_handler(err);
  return 0;
}

ypic
yw_frame(ywin w)
{
  ypic failure = {0};

  XWindowAttributes attrs;
  /* So far I haven’t seen this fail, but in theory it can; presumably
   * if the window had been destroyed, for example */
  if (!XGetWindowAttributes(w->display, w->window, &attrs)) {
    yeso_error_handler("XGetWindowAttributes");
    return failure;
  }

  /* We might have no shared-memory framebuffer or one with a
   *  different size; in those cases we need to make a new one. */
  if (!w->image || w->width != attrs.width || w->height != attrs.height) {
    if (!setup_framebuffer(w, attrs.width, attrs.height)) return failure;
  }

  // This check saves you from the common SDL bug where you draw a
  // bunch of stuff and nothing shows up on the screen because you
  // never called flip():
  if (w->valid) {
    yeso_error_handler("yw_flip wasn’t called");
    return failure;
  }

  w->valid = 1;
  ypic result = { (ypix*)w->shm_segment_info.shmaddr, w->width, w->width, w->height };
  return result;
}

static void
mouse_event(yw_event *ev, int x, int y, int state)
{
  ev->type = yw_mouse;
  /* button 1 is 0x100, 2 is 0x200, 3 is 0x400, 4 (wheel up) 0x800,
     5 (wheel down) 0x1000 */
  yw_mouse_event mev = { .x = x, .y = y, .buttons = (state >> 8) & 0x1f };
  ev->body.mouse = mev;
}

/** Read an event from X (in blocking mode) and translate it to a `yw_event`
 *
 * This is invoked both from `yw_wait` (since that’s the way X
 * gives us to wait for an event) and from `yw_get_event` (when
 * there isn’t an event waiting for us from `yw_wait`.)
 */
static void
process_pending_event(ywin w)
{
  yw_event *ev = &w->event;
  memset(ev, 0, sizeof(w->event)); /* sets event type to yw_raw */
  ev->w = w;

  XNextEvent(w->display, &w->xevent);
  int type = w->xevent.type;

  if (type == MotionNotify) {
    XMotionEvent *mev = &w->xevent.xmotion;
    mouse_event(ev, mev->x, mev->y, mev->state);

  } else if (type == ButtonPress || type == ButtonRelease) {
    XButtonEvent *bev = &w->xevent.xbutton;
    mouse_event(ev, bev->x, bev->y, bev->state);
    ev->body.mouse.buttons ^= 1 << (bev->button - 1);

  } else if (type == KeyPress || type == KeyRelease) {
    XKeyEvent *kev = &w->xevent.xkey;
    yw_key_event ykev = { .s = &w->keybuf[0], .down = (type == KeyPress) };
    KeySym keysym;
    ykev.len = XLookupString(kev, ykev.s, sizeof(w->keybuf)-1, &keysym, NULL);
    ykev.s[ykev.len] = 'E';
    ykev.keysym = keysym;
    if (
        kev->state & ControlMask &&  // If ctrl is held
        (ykev.keysym != yk_control_l && ykev.keysym != yk_control_r)) 
    {
        ykev.keysym |= 1ull<<32; // Act like it's a special new key
        ykev.len += 2;
        strcpy(ykev.s+2, ykev.s);
        ykev.s[0]='C';
        ykev.s[1]='-';
    }
    ev->type = yw_key;
    ev->body.key = ykev;

  } else if (type == ClientMessage) {
    XClientMessageEvent *cm = &w->xevent.xclient;
    if (cm->message_type == w->WM_PROTOCOLS
        && cm->format == 32     /* 32-bit words */
        && cm->data.l[0] == w->WM_DELETE_WINDOW) {
      ev->type = yw_die;
    }

  } else {
    ev->type = yw_raw;
    ev->body.raw = &w->xevent;
  }
}

void
yw_wait(ywin w, long timeout_usecs)
{
  /* Note that calling this twice in a row without calling
     `yw_get_event` in between will drop events.  This is
     intentional. */
  if (timeout_usecs && !XPending(w->display)) {
    struct timeval tv = { .tv_sec = 0, .tv_usec = timeout_usecs };
    fd_set fds;
    FD_ZERO(&fds);
    int fd = ConnectionNumber(w->display);
    FD_SET(fd, &fds);
    if (!select(fd+1, &fds, NULL, NULL, &tv)) return;
    /* XXX what if we got data on the file descriptor but not an
       entire X event, so that process_pending_event below will still
       hang?  This is probably a bug that should be fixed */
  }
  process_pending_event(w);
  w->has_event = 1;
}

yw_event *
yw_get_event(ywin w)
{
  if (w->has_event) {
    w->has_event = 0;
    return &w->event;
  }

  if (!XPending(w->display)) {
    return NULL;
  }

  // Though this is a blocking call, XPending returned true, so it’s
  // guaranteed not to block now.  I hope.
  process_pending_event(w);
  return &w->event;
}

yw_mouse_event *
yw_as_mouse_event(yw_event *ev)
{
  return (ev && ev->type == yw_mouse) ? &ev->body.mouse : NULL;
}

yw_key_event *
yw_as_key_event(yw_event *ev)
{
  return (ev && ev->type == yw_key) ? &ev->body.key : NULL;
}

yw_raw_event *
yw_as_raw_event(yw_event *ev)
{
  return (ev && ev->type == yw_raw) ? &ev->body.raw : NULL;
}

yw_die_event
yw_as_die_event(yw_event *ev)
{
  /* If the client never calls this function, we presume that it
     doesn't know how to handle die events, and consequently we don’t
     request them from the X window manager; this allows simple
     output-only clients like yesomunch to live in blissful ignorance
     of all events of any kind.  So, the first time the client does
     call it, we use XSetWMProtocols to request them.  Normally this
     happens pretty early in the lifecycle of a window, but it does
     happen after the window is initially mapped (and reparented and
     whatnot).  It’s possible that some window manager out there will
     have trouble with this — in fact, as with everything in fucking
     ICCCM, I'm almost certain of it — but at least marco, the MATE
     window manager, handles it fine.
   */
  if (!ev) return 0;
  ywin w = ev->w;
  if (!w->handles_die) {
    if (!XSetWMProtocols(w->display, w->window, &w->WM_DELETE_WINDOW, 1)) {
      yeso_error_handler("XSetWMProtocols");
      return 0;
    }

    w->handles_die = 1;
  }

  return ev->type == yw_die;
}

void
yw_flip(ywin w)
{
  if (!w->valid) {
    yeso_error_handler("yw_frame wasn’t called");
    return;
  }

  XShmPutImage(w->display, w->window, w->gc, w->image,
               0, 0, 0, 0,
               w->width, w->height,
               False);
  XSync(w->display, False);
  // Cabbage the framebuffer to force clients to redraw.
  // This is costly: it costs 300+ μs at 640×480.  So I may remove it.
  memset(w->shm_segment_info.shmaddr, 192, 4 * w->width * w->height);
  // Force the clients to call `yw_frame` again before
  // redrawing.
  w->valid = 0;
}

void
yw_fds(ywin w, void (*callback)(void *, int fd), void *userdata)
{
  callback(userdata, ConnectionNumber(w->display));
}

#ifndef __GNUC__
#define __attribute__(x)
#endif

int __attribute__((weak))
main(int argc, char **argv)
{
  ywin w = yw_open("yeso test", 640, 480, "");
  int xorg = 0, yorg = 0;

  struct timeval start;
  gettimeofday(&start, NULL);
  int frames = 0;
  for (;;frames++) {
    struct timeval now;
    gettimeofday(&now, NULL);
    long us = (now.tv_sec - start.tv_sec) * 1000L*1000 +
      (now.tv_usec - start.tv_usec);

    for (yw_event *ev; (ev = yw_get_event(w));) {
      if (yw_as_die_event(ev)) {
        yw_close(w);
        return 0;
      }

      yw_mouse_event *mev = yw_as_mouse_event(ev);
      if (mev) {
        xorg = mev->x;
        yorg = mev->y;
      }
    }

    ypic fb = yw_frame(w);
    for (int y = 0; y < fb.h; y++) {
      int dy = y - yorg;
      for (int x = 0; x < fb.w; x++) {
        int dx = x - xorg;
        *yp_pix(fb, x, y) = dx*dx + dy*dy - us / 4096;
      }
    }

    yw_flip(w);
    if (us && !(frames & 0xff)) {
      /* This is showing me 370 fps on my laptop, or 470 fps if I
         remove the call to memset the frame buffer in yw_flip, or 92
         fps at 1920×1080. */
      printf("%d frames in %ld μs = %.1f fps\n",
             frames, us, ((double)frames/us)*1e6);
    }
  }
}
