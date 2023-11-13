# SigmaZero
Implementation of AlphaZero as part of the course project in CS337.

## Compiling shared object file
`g++ -O3 CPP_backend.cpp -shared -fpic -Wno-undef -I /usr/include/python3.10/ -L /usr/lib/x86_64-linux-gnu/  -lboost_python310 -lboost_system -lboost_numpy310 -o CPP_backend.so`