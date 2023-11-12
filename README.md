A simple raytracer written in C.

Based on a [raytracer](http://canonical.org/~kragen/sw/aspmisc/raytracer.c) written written by [kragen](http://canonical.org/~kragen/sw/aspmisc/my-very-first-raytracer.html); and on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) by Peter Shirley, Trevor David Black, Steve Hollasch.

See a [video demo](https://youtu.be/y4TcrxRg4aw).

## Raytracing Progress
---
(Day 03): Write basic raytracer

Fix reference image. Initial render takes 14.1s.
Add CUDA support. Base render is now 621ms (1.6 fps) with no changes except to RNG.

Reduce max bounces to 5, and move pixel byte computation to the GPU.
