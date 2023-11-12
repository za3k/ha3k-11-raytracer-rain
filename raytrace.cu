#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C" {
#include "yeso.h"
}

#define W 400
#define H 300
#define MAX_OBJECTS 3000
#define SAMPLES 1
#define MAX_BOUNCES 5
#define PIXELS (W*H)
#define THREADS 256
#define BLOCKS (ceil(PIXELS * 1.0) / THREADS)
#define OUTPUT_VIDEO 1
// Only affects the video file output--rendered output is always as fast as possible.
#define FRAMERATE 60

#define ZOOM 1

/* Types */
typedef double sc; // scalar
typedef struct { sc x, y, z; } vec;
typedef struct { unsigned char b, g, r, a; } pix;

/* Vectors */
__host__ __device__ inline static sc dot(vec aa, vec bb)   { return aa.x*bb.x + aa.y*bb.y + aa.z*bb.z; }
__host__ __device__ inline static sc magsq(vec vv)         { return dot(vv, vv); }
__host__ __device__ inline static vec scale(vec vv, sc c)  { vec rv = { vv.x*c, vv.y*c, vv.z*c }; return rv; }
__device__ inline static vec normalize(vec vv)    { return scale(vv, rnorm3d(vv.x, vv.y, vv.z)); }
__host__ inline static vec h_normalize(vec vv)    { return scale(vv, 1.0/sqrt(magsq(vv))); }
__host__ __device__ inline static vec add(vec aa, vec bb)  { vec rv = { aa.x+bb.x, aa.y+bb.y, aa.z+bb.z }; return rv; }
__device__ inline static vec sub(vec aa, vec bb)  { return add(aa, scale(bb, -1)); }
__device__ inline static vec hadamard_product(vec aa, vec bb) { vec rv = { aa.x*bb.x, aa.y*bb.y, aa.z*bb.z }; return rv; }

/* Ray-tracing types */
typedef vec color;              // So as to reuse dot(vv,vv) and scale
typedef struct { color albedo; sc reflectivity; sc fuzz; } material;
typedef struct { vec cp; material ma; sc r; vec vel; vec acc; } sphere;
typedef struct { sphere spheres[MAX_OBJECTS]; int nn; } world;
typedef struct { vec start; vec dir; } ray; // dir is normalized!

/* Random sampling */

__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__host__ static sc random_double() { return (rand() / (RAND_MAX + 1.0)); } // [0, 1)
__host__ static sc random_double(double min, double max) { return (max-min)*random_double() + min; } // [min, max)
__host__ static color random_color() { vec v = { random_double(), random_double(), random_double() }; return v; }

__device__ static sc d_random_double(curandState *d_randstate) { return curand_uniform_double(d_randstate); }
__device__ static vec d_random_vec(curandState *d_randstate) { vec v = { d_random_double(d_randstate), d_random_double(d_randstate), d_random_double(d_randstate) }; return v; }
__device__ static vec d_random_in_unit_sphere(curandState *d_randstate) {
    while (1) {
        vec v = d_random_vec(d_randstate);
        if (magsq(v) <= 1) return v;
    }
}
__device__ static vec d_random_unit_vector(curandState *d_randstate) { return normalize(d_random_in_unit_sphere(d_randstate)); }

/* Ray-tracing */

__device__ static color BLACK = {0,    0,    0  };
__device__ static color WHITE = {1.0,  1.0,  1.0};
__device__ static color BLUE =  {0.25, 0.49, 1.0};

__device__ static vec reflect(vec incoming, vec normal) {
    return sub(incoming, scale(normal, dot(incoming,normal)*2));
}

__device__ static int find_nearest_intersection(ray rr, sphere ss, sc *intersection) {
  vec center_rel = sub(rr.start, ss.cp);
  // Quadratic coefficients of parametric intersection equation.  a == 1.
  sc half_b = dot(center_rel, rr.dir);
  sc c = magsq(center_rel) - ss.r*ss.r;
  sc discrim = half_b*half_b - c;
  if (discrim < 0) return 0;
  sc sqdiscrim = sqrt(discrim);
  *intersection = (-half_b - sqdiscrim > 0 ? (-half_b - sqdiscrim)
                                           : (-half_b + sqdiscrim));
  return 1;
}

__device__ static color ray_color(curandState *randstate, const world *here, ray rr)
{
  color albedo = WHITE;

  for (int depth = 0; depth < MAX_BOUNCES; depth++) {
    const sphere *nearest_object = 0;
    sc nearest_t = 1/.0;
    sc intersection;

    for (int i = 0; i < here->nn; i++) {
      if (find_nearest_intersection(rr, here->spheres[i], &intersection)) {
        if (intersection < 0.00001 || intersection >= nearest_t) continue;
        nearest_t = intersection;
        nearest_object = &here->spheres[i];
      }
    }

    if (!nearest_object) {
        // Sky color
        sc a = 0.5 * (rr.dir.y + 1);
        return hadamard_product(albedo, add(scale(WHITE, 1.0-a), scale(BLUE, a)));
    }

    // Object color
    vec point = add(rr.start, scale(rr.dir, nearest_t));
    vec normal = normalize(sub(point, nearest_object->cp));
    vec dir = d_random_unit_vector(randstate);

    ray bounce = { point };
    if (nearest_object->ma.reflectivity == 0) { // Matte, regular scattering
      bounce.dir = add(normal, dir);
    } else { // Reflective metal scattering
      vec reflected = reflect(rr.dir, normal);
      bounce.dir = add(reflected, scale(dir, nearest_object->ma.fuzz * 0.99999));
    }
    bounce.dir = normalize(bounce.dir);
    rr = bounce;
    albedo = hadamard_product(albedo, nearest_object->ma.albedo);
  }
  return BLACK;
}

/* Rendering */

__device__ static unsigned char
byte(double dd) { return dd > 1 ? 255 : dd < 0 ? 0 : dd * 255 + 0.5; }

__device__ static ray get_ray(curandState *randstate, int x, int y) {
  // Camera is always at 0,0
  sc aspect = ((sc)W)/H; // Assume aspect >= 1
  sc viewport_height = 2.0;
  sc focal_length = 1.0; // Z distance of viewport
  sc viewport_width = viewport_height * aspect;

  sc pixel_width = (viewport_width / W);
  sc pixel_height = (viewport_height / H);
  sc left = viewport_width / -2.0;
  sc top = viewport_height / 2.0;

  sc px = left + (pixel_width * (x + d_random_double(randstate)));
  sc py = top - (pixel_height * (y + d_random_double(randstate)));

  vec pv = { px, py, focal_length };
  ray rr = { {0}, normalize(pv) };

  return rr;
}

__device__ static pix render_pixel(curandState *randstate, const world *here, int x, int y)
{
  color pixel_color = {0, 0, 0};
  for (int sample = 0; sample < SAMPLES; ++sample) {
    ray rr = get_ray(randstate, x, y);
    pixel_color = add(pixel_color, ray_color(randstate, here, rr));
  }
  pixel_color = scale(pixel_color, 1.0/SAMPLES);
  pix p = { .b = byte(sqrt(pixel_color.z)), .g = byte(sqrt(pixel_color.y)), .r = byte(sqrt(pixel_color.x)), .a = 0, }; 
  return p;
}

__global__ void render_pixels(curandState *randstate, const world *here, pix *result)
{
  // COPY world + randstate

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  int x = idx % W;
  int y = idx / W;

  curandState state = randstate[threadIdx.x];

  if (idx < PIXELS) {
    result[y*W+x] = render_pixel(&state, here, x, y);
  }
}

static void render(curandState *d_randstate, world *h_here, ypic fb, FILE *rgb24_stream)
{
  // Copy the world to the GPU
  static world *d_here = 0;
  if (d_here == 0) cudaMalloc(&d_here, sizeof(world));
  cudaMemcpy(d_here, h_here, sizeof(world), cudaMemcpyHostToDevice);

  // Allocate space for the result
  static pix *d_result = 0;
  if (d_result == 0) cudaMalloc(&d_result, sizeof(pix)*PIXELS);

  // Calculate the pixels
  render_pixels<<<BLOCKS, THREADS>>>(d_randstate, d_here, d_result);
  static pix *h_result;
  if (h_result == 0) h_result = (pix *)malloc(sizeof(color)*PIXELS);
  cudaMemcpy(h_result, d_result, PIXELS * sizeof(pix), cudaMemcpyDeviceToHost);

  // Render to yeso
  for (int yy=0; yy<H; ++yy) {
    for (int ii=0; ii<ZOOM; ++ii) {
      for (int xx=0; xx<W; ++xx) {
        for (int jj=0; jj<ZOOM; ++jj) {
          pix *p = &h_result[yy*W+xx];
          ypix *r = yp_pix(fb, ZOOM*xx+jj, ZOOM*yy+ii);
          *r = (p->b) | (p->g << 8) | (p->r << 16);
        }
      }
    }
  }

  // Render to stream
  if (rgb24_stream) {
    for (int yy=0; yy<H; ++yy) {
      for (int xx=0; xx<W; ++xx) {
        pix *p = &h_result[yy*W+xx];
        fputc(p->r, rgb24_stream);
        fputc(p->g, rgb24_stream);
        fputc(p->b, rgb24_stream);
      }
    }
  }
}

// Ground
sphere ground  = { .cp = {0, -1005, 0}, .ma = { .albedo = {0.5, 0.5, 0.5}, .reflectivity = 1.0 }, .r = 1000 };


void deleteObject(world *here, int i) {
    memcpy(&here->spheres[i], &here->spheres[i+1], sizeof(sphere) * (here->nn - i));
    here->nn--;
}

void addObject(world *here, sphere *s) {
    if (here->nn >= MAX_OBJECTS) return;
    memcpy(&here->spheres[here->nn], s, sizeof(sphere));
    here->nn++;
}

sphere* createRandom() {
    static sphere ran;
    ran.r = random_double(0.6, 1.2);
    ran.cp.x = random_double(-10.0, 10.0);
    ran.cp.z = random_double(5.0, 10.0);
    ran.cp.y = random_double(10.0, 30.0);
    ran.vel.x = ran.vel.z = 0.0;
    ran.acc.y = -0.015;
    ran.acc.x = ran.acc.z = 0;
    ran.acc.y = -0.001;
    ran.ma.albedo = random_color();
    if (random_double() > 0.8) {
        ran.ma.reflectivity = 1;
        ran.ma.fuzz = random_double(0, 0.5);
    }

    return &ran;
}

int isSmall(sphere *s) {
    return (s->r <= 0.5);
}

sphere* createDroplet(sphere *s) {
    static sphere drop;
    drop.r = random_double(0.1, 0.2);
    drop.cp = s->cp;
    drop.cp.y -= 0.9 * s->r;
    drop.vel = s->vel;
    drop.vel.y = random_double(0, -drop.vel.y * 0.8);
    double SPEED = 0.04;

    drop.vel.x += random_double(-SPEED, SPEED);
    drop.vel.z += random_double(-SPEED, SPEED);
    drop.acc = s->acc;
    drop.ma.reflectivity = s->ma.reflectivity;
    drop.ma.albedo = add(scale(random_color(), 0.5), s->ma.albedo);
    if (magsq(drop.ma.albedo) > 1.0) drop.ma.albedo = h_normalize(drop.ma.albedo);

    return &drop;
}
void explode(world *here, sphere *s) {
    for (int i = 0; i<20; i++) addObject(here, createDroplet(s));
}
int inWorld(sphere *s) {
    if (s->r > 500) return true; // Ground is always in the world.
    else if (isSmall(s)) return (s->cp.y + s->r < 5.1); // Underwater?
    else return (s->cp.y - s->r > -5.1); // Touching water?
}

void tick(world *here) {
  for (int i=0; i<here->nn; ++i) {
    sphere *s = &here->spheres[i];
    if (inWorld(s)) {
        // Move
        s->vel = add(s->acc, s->vel);
        s->cp = add(s->cp, s->vel);
    } else {
        if (!isSmall(s)) {
            explode(here, s);
            addObject(here, createRandom());
        }
        deleteObject(here, i);
    }
  }
  while (here->nn < 10) addObject(here, createRandom());
}
void scene(world *here) {
    addObject(here, &ground);
}

int main(int argc, char **argv) {
  // Set up CUDA
  curandState *d_randstate;
  cudaMalloc(&d_randstate, sizeof(curandState)*THREADS);
  setup_kernel<<<1, 1024>>>(d_randstate);

  // Set up world
  static world here = {0};
  scene(&here);

  // Set up yeso
  ywin w = yw_open("raytracer in yeso and CUDA", W*ZOOM, H*ZOOM, "");

  // Set up video output (via ffmpeg)
  char CMD[1000];
  char OUTFILE[100];
  sprintf(OUTFILE, "raytrace-%dx%d-%d_sample.mkv", W, H, SAMPLES);
  sprintf(CMD, "ffmpeg -f rawvideo -pix_fmt rgb24 -r %d -s:v %dx%d -loglevel 16 -i - -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv444p -y -- %s", FRAMERATE, W, H, OUTFILE);
  FILE *stream = OUTPUT_VIDEO ? popen(CMD, "w") : 0;

  // Set up fps timer
  clock_t start = clock(), stop;

  for (int frame=1; frame<1000; ++frame) {
    tick(&here);
    ypic fb = yw_frame(w);
    render(d_randstate, &here, fb, stream);
    yw_flip(w);
    stop = clock();
    int us = (stop-start)*1.0 / frame;
    if (!(frame & 0xff))
      fprintf(stderr, "Render: %0.1f fps (%ldms for frame %d)\n", 1000000.0/us, us/1000, frame);
  }

  fclose(stream);
  return 0;
}
