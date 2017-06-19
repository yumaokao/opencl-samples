#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <CL/cl.hpp>
#include <boost/program_options.hpp>

const char * addoneStr = "__kernel void "
                        "addone(global const int* A, global int* C) "
                        "{ "
                        "  C[get_global_id(0)] = A[get_global_id(0)] + 1;"
                        "} ";

namespace po = boost::program_options;

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
        int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
        cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
        queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
        
        // Programs
        cl::Program::Sources source(1, std::make_pair(addoneStr, strlen(addoneStr)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        // Kernel
        cl::Kernel kernel(program_, "addone", &err);
        kernel.setArg(0, buffer_A);
        kernel.setArg(1, buffer_C);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(10), cl::NullRange, NULL, &event);
        queue.finish();

        // Check
        int C[10];
        queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);
        std::cout << "result: " << std::endl;
        for (int i = 0; i < 10; i++)
            std::cout << C[i] << " ";
        std::cout << std::endl;
    }
    catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
