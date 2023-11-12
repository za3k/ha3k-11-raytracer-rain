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
#define MAX_OBJECTS 30
#define SAMPLES 1
#define MAX_BOUNCES 5
#define PIXELS (W*H)
#define THREADS 256
#define BLOCKS (ceil(PIXELS * 1.0) / THREADS)
#define OUTFILE "raytrace-200x150.mkv"
#define OUTPUT_VIDEO 1
// Only affects the video file output--rendered output is always as fast as possible.
#define FRAMERATE 60

#define ZOOM 4

/* Types */
typedef double sc; // scalar
typedef struct { sc x, y, z; } vec;
typedef struct { unsigned char b, g, r, a; } pix;

/* Vectors */
__device__ inline static sc dot(vec aa, vec bb)   { return aa.x*bb.x + aa.y*bb.y + aa.z*bb.z; }
__device__ inline static sc magsq(vec vv)         { return dot(vv, vv); }
__device__ inline static vec scale(vec vv, sc c)  { vec rv = { vv.x*c, vv.y*c, vv.z*c }; return rv; }
__device__ inline static vec normalize(vec vv)    { return scale(vv, rnorm3d(vv.x, vv.y, vv.z)); }
__device__ inline static vec add(vec aa, vec bb)  { vec rv = { aa.x+bb.x, aa.y+bb.y, aa.z+bb.z }; return rv; }
__device__ inline static vec sub(vec aa, vec bb)  { return add(aa, scale(bb, -1)); }
__device__ inline static vec hadamard_product(vec aa, vec bb) { vec rv = { aa.x*bb.x, aa.y*bb.y, aa.z*bb.z }; return rv; }

/* Ray-tracing types */
typedef vec color;              // So as to reuse dot(vv,vv) and scale
typedef struct { color albedo; sc reflectivity; sc fuzz; } material;
typedef struct { vec cp; material ma; sc r; } sphere;
typedef struct { sphere spheres[MAX_OBJECTS]; int nn; } world;
typedef struct { vec start; vec dir; } ray; // dir is normalized!

/* Random sampling */

__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__host__ static sc random_double() { return (rand() / (RAND_MAX + 1.0)); } // [0, 1)
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
  world *d_here;
  cudaMalloc(&d_here, sizeof(world));
  cudaMemcpy(d_here, h_here, sizeof(world), cudaMemcpyHostToDevice);

  // Allocate space for the result
  pix *d_result;
  cudaMalloc(&d_result, sizeof(pix)*PIXELS);

  // Calculate the pixels
  render_pixels<<<BLOCKS, THREADS>>>(d_randstate, d_here, d_result);
  pix *h_result = (pix *)malloc(sizeof(color)*PIXELS);
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
sphere ground  = { .cp = {0,  -1000, 5}, .ma = { .albedo = {0.5, 0.5, 0.5} }, .r = 1000 };
// Sphere 1, reflective (fuzzier)
sphere sphere1 = { .cp = {-2, 1.0,   5}, .ma = { .albedo = {0.7, 0.7, 0.7}, .reflectivity = 1.0, .fuzz = 0.3 }, .r = 1 };
// Sphere 2, matte brown
sphere sphere2 = { .cp = {0,  1.0, 5},   .ma = { .albedo = {0.4, 0.2, 0.1} }, .r = 1 };
// Sphere 3, reflective
sphere sphere3 = { .cp = {2,  1.0, 5},    .ma = { .albedo = {0.5, 0.5, 0.5}, .reflectivity = 1.0, }, .r = 1 };
void scene(world *here) {
  sc ALT = -2.0;
  sc RAD = 0.2;

  here->spheres[here->nn++] = ground;
  here->spheres[here->nn++] = sphere1;
  here->spheres[here->nn++] = sphere2;
  here->spheres[here->nn++] = sphere3;

  for (int a=-2; a<=2; a++) {
    for (int b=3; b<=7; b++) {
      // Add a sphere
      sphere *s = &here->spheres[here->nn++];
      s->cp.x = a + 0.9*random_double();
      s->cp.y = RAD;
      s->cp.z = b + 0.9*random_double();
      s->r = RAD;
      s->ma.reflectivity = random_double() > 0.8;
      s->ma.fuzz = random_double();
      s->ma.albedo = random_color();
    }
  }

  for (int i=0; i<here->nn; i++) here->spheres[i].cp.y += ALT;
}

int main(int argc, char **argv) {
  // Set up CUDA
  curandState *d_randstate;
  cudaMalloc(&d_randstate, sizeof(curandState)*THREADS);
  setup_kernel<<<1, 1024>>>(d_randstate);

  // Set up world
  world here = {0};
  scene(&here);

  // Set up yeso
  ywin w = yw_open("raytracer in yeso and CUDA", W*ZOOM, H*ZOOM, "");

  // Set up video output (via ffmpeg)
  char CMD[1000];
  sprintf(CMD, "ffmpeg -f rawvideo -pix_fmt rgb24 -r %d -s:v %dx%d -loglevel 16 -i - -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv444p -y -- %s", FRAMERATE, W, H, OUTFILE);
  FILE *stream = OUTPUT_VIDEO ? popen(CMD, "w") : 0;

  // Set up fps timer
  clock_t start = clock(), stop;

  for (int frame=1; frame <= 600; ++frame) {
    ypic fb = yw_frame(w);
    render(d_randstate, &here, fb, stream);
    yw_flip(w);
    for (int i=0; i<here.nn; i++) {
      here.spheres[i].cp.y -= 0.005;
    }
    stop = clock();
    int us = (stop-start)*1.0 / frame;
    if (!(frame & 0xff))
      fprintf(stderr, "Render: %ldms (%0.1f fps)\n", us/1000, 1000000.0/us);
  }

  fclose(stream);
  return 0;
}
