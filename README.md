Cool [rain](https://youtu.be/FewtsNn8dg0) demo of a simple raytracer in action.

This simple simple raytracer written in C and CUDA. It's based on a [raytracer](http://canonical.org/~kragen/sw/aspmisc/raytracer.c) written written by [kragen](http://canonical.org/~kragen/sw/aspmisc/my-very-first-raytracer.html); and on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) by Peter Shirley, Trevor David Black, Steve Hollasch.

This demo has 10-3000 objects on screen at a time. 10 main drops, plus tinier droplets that bounce off.

The below rendering times used a 4060Ti NVidia GPU.

Resolution | Samples   | Framerate
==================================
400x300    | 1/pixel   | 81.7 fps
400x300    | 10/pixel  | 20.6 fps
400x300    | 100/pixel | 2.4 fps
1920x1080  | 1/pixel   | 5.4 fps
1920x1080  | 10/pixel  | 1.4 fps
1920x1080  | 100/pixel | 0.2 fps
