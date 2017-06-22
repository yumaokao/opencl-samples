#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>

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

    return EXIT_SUCCESS;
}
// vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
