# SigmaZero
Implementation of AlphaZero as part of the course project in CS337.

## Instructions to Compile C++ backend

- Install Boost library from [here](https://www.boost.org/doc/libs/1_46_1/more/getting_started/unix-variants.html)
- To compile C++ backend into a shared object file use : `g++ -O3 CPP_backend.cpp -shared -fpic -Wno-undef -I BOOST_INSTALL_PATH -I /usr/include/python3.x/ -L BOOST_INSTALL_PATH/stage/lib/  -lboost_python3x -lboost_system -o CPP_backend.so`
- Replace `x` with appropriate python version number installed in your machine. Python 3.10 is recommended.

## Model Weights

[Link](https://drive.google.com/file/d/12QiXUTJTqZ05LSDosasyz2efBJ46E7CS/view?usp=sharing) to the weights of the final model.
