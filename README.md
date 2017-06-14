# opencl-samples

# Install OpenCL

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
