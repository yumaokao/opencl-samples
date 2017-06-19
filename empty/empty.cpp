#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <CL/cl.hpp>
#include <boost/program_options.hpp>

const char * emptyStr = "__kernel void "
                        "empty(void) "
                        "{ "
                        "  "
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
        ("platform,p", po::value<int>(&pid)->default_value(0), "platform id")
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

        cl::Program::Sources source(1, std::make_pair(emptyStr,strlen(emptyStr)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        cl::Kernel kernel(program_, "empty", &err);

        cl::Event event;
        cl::CommandQueue queue(context, devices[0], 0, &err);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(4,4), cl::NullRange, NULL, &event);

        queue.finish();
        // event.wait();
    }
    catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
