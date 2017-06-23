#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <caffe/proto/caffe.pb.h>
#include <CL/cl.hpp>

namespace po = boost::program_options;

const char * addoneStr = "__kernel void "
                        "addone(global const float* A, global float* C) "
                        "{ "
                        "  C[get_global_id(0)] = A[get_global_id(0)] + 1.0;"
                        "} ";

int main(int argc, char *argv[])
{
    int pid;
    int verbose;
    po::options_description desc(std::string (argv[0]).append(" options"));
    desc.add_options()
        ("help,h", "produce help message")
        ("verbose,v", po::value<int>(&verbose)->default_value(0), "verbose level")
        ("platform,p", po::value<int>(&pid)->default_value(1), "platform id")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    if (verbose > 1)
        std::cout << "platform id = " << pid << std::endl;

    // ifstream
    std::ifstream *bfile = nullptr;
    bfile = new std::ifstream("conv2.bin", std::ios::in | std::ios::binary);

    // read blob
    caffe::BlobProto blob;
    if (bfile->is_open()) {
        blob.ParseFromIstream(bfile);
        if (blob.has_shape()) {
            caffe::BlobShape shape = blob.shape();
            std::cout << "dim_size() " << shape.dim_size() << std::endl;
            for (int i = 0; i < shape.dim_size(); i++)
                std::cout << "  dim[" << i << "] " << shape.dim(i) << std::endl;
        }

        std::cout << "data_size() " << blob.data_size() << std::endl;
        // const float *fdata = nullptr;
        // fdata = blob.data().data();
        bfile->close();
    }

    // OpenCL addone
    cl_int err = CL_SUCCESS;
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cerr << "No OpenCL Platform" << std::endl;
            return -1;
        }
        if (pid > platforms.size() - 1) {
            std::cerr << "Invalid Platform ID" << std::endl;
            return -1;
        }
        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pid])(), 0};
        cl::Context context(CL_DEVICE_TYPE_ALL, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0], 0, &err);

        // Buffers
        size_t elemsize = blob.data_size();
        cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * elemsize);
        cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float) * elemsize);
        queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * elemsize, blob.data().data());

        // Programs
        cl::Program::Sources source(1, std::make_pair(addoneStr, strlen(addoneStr)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        // Kernel
        cl::Kernel kernel(program_, "addone", &err);
        kernel.setArg(0, buffer_A);
        kernel.setArg(1, buffer_C);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(elemsize), cl::NullRange, NULL, &event);
        queue.finish();

        // Check
        const float *A = nullptr;
        A = blob.data().data();
        std::cout << "A: " << std::endl;
        for (int i = 0; i < 128; i++) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << A[i] << " ";
            if ((i + 1) % 16 == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;

        float C[sizeof(float) * elemsize];
        queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * elemsize, C);
        std::cout << "C: " << std::endl;
        for (int i = 0; i < 128; i++) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << C[i] << " ";
            if ((i + 1) % 16 == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return EXIT_SUCCESS;

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
