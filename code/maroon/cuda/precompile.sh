for f in *.cu; do
    nvcc -Xptxas -O3 -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -cubin --use_fast_math -dopt=on $f
done