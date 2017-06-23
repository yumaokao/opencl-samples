#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <caffe/proto/caffe.pb.h>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    int verbose;
    po::options_description desc(std::string (argv[0]).append(" options"));
    desc.add_options()
        ("help,h", "produce help message")
        ("verbose,v", po::value<int>(&verbose)->default_value(0), "verbose level")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    // ifstream
    std::ifstream *bfile = nullptr;
    bfile = new std::ifstream("conv2.bin", std::ios::in | std::ios::binary);

    // read blob
    const float *fdata = nullptr;
    if (bfile->is_open()) {
        caffe::BlobProto blob;
        blob.ParseFromIstream(bfile);
        if (blob.has_shape()) {
            caffe::BlobShape shape = blob.shape();
            std::cout << "dim_size() " << shape.dim_size() << std::endl;
            for (int i = 0; i < shape.dim_size(); i++)
                std::cout << "  dim[" << i << "] " << shape.dim(i) << std::endl;
        }

        std::cout << "data_size() " << blob.data_size() << std::endl;
        fdata = blob.data().data();

        bfile->close();
    }

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
