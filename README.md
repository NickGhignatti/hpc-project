# HPC Course project

## OpenMP version

```bash
To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp ompcircles.c -o ompcircles -lm

To execute:

        ./ompcircles [ncircles [iterations]]

where `ncircles` is the number of circles, and `iterations` is the
number of iterations to execute.

If you want to produce a movie (this is not required, and should be
avoided when measuring the performance of the parallel versions of
this program) compile with:

        gcc -std=c99 -Wall -Wpedantic -DMOVIE -fopenmp ompcircles.c -o ompcircles -lm

and execute with:

        ./ompcircles.movie 200 500

A lot of `circles-xxxxx.gp` files will be produced; these files must
be processed using `gnuplot` to create individual frames:

        for f in *.gp; do gnuplot "$f"; done

and then assembled to produce the movie `circles.avi`:

        ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi
```

## CUDA version

```bash
To compile:

        nvcc cudacircles.cu -o cudacircles -lm

To execute:

        ./cudacircles [ncircles [iterations]]

where `ncircles` is the number of circles, and `iterations` is the
number of iterations to execute.
```