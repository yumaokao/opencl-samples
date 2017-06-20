# opencl-samples

## Install OpenCL
On Arch Linux, install opencl libraries with yaourt:

```sh
$ yaourt beignet
$ yaourt amdapp-sdk
```

Then could check OpenCL platforms with `clinfo`:

```sh
$ yaourt clinfo
$ clinfo
Number of platforms                               2
  Platform Name                                   Intel Gen OCL Driver
  ...

  Platform Name                                   AMD Accelerated Parallel Processing
  ...
```

## Build and Run
Build with cmake

```sh
$ mkdir build
$ cd build
$ cmake ../
$ make
```

Run
```sh
# run empty
$ ./empty

# run simple add one, should see
# result:
# 1 2 3 4 5 6 7 8 9 10
$ ./addone
```

## Issues
1. drm_intel_gem_bo_context_exec() failed: No space left on device
  On my Haswell, (Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz), beigent 1.3 is reported this and failed.
  However in other Broadwell, everything is fine.
