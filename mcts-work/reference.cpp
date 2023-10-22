#include <iostream>
#include <Python.h>
#include <boost/python.hpp>

// g++ $(python3-config --cflags) -fPIE -c test.cpp -o test.o
// g++ test.o -L /usr/lib/x86_64-linux-gnu/ -lboost_system -lboost_python310 $(python3-config --embed --ldflags) -o test

using namespace std;
namespace python = boost::python;

int main(int argc, char* argv[]) {
  Py_Initialize();  
  python::object sample_module = python::import("chess");
  python::object result = sample_module.attr("STARTING_FEN");
//   python::object result = sample_function(5.0);
  std::string y = python::extract<std::string>(result);
  cout << y << endl;
  return 0;
}
