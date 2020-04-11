The simple implementation of the grid topology verification kernel can be found
in the grid_debug.cu
In principle you can start a new project using nsight and compile it within the
RAD, but just for fun and to get a bit more hands on please copy it to your work
area and compile by hand:

nvcc grid_debug.cu -o grid_debug

and run it:

./grid_debug

