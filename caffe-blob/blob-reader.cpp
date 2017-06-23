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
        const float *fdata = nullptr;
        fdata = blob.data().data();

        // for conv2.bin transposed to NHWC, padded to 8 elements
        //   nhwc(0, 3, 5, 1) = 24.6118
        //   nhwc(0, 0, 7, 0) = 37.1379
        std::cout << "float data(0, 3, 5, 1) " << fdata[(1) + (5) * 256 + (3) * 256 * 32] << std::endl;
        std::cout << "float data(0, 0, 7, 0) " << fdata[(0) + (7) * 256 + (0) * 256 * 32] << std::endl;
        bfile->close();
    }

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
