/*
Compilation:
  g++ wrapping_cpp_function.cpp -shared -fPIC -O3 -o my_lib.so

Python code to call the c function:

import numpy as np
import ctypes

x = np.array(np.random.normal(0,1,100),dtype=np.float32)
y = np.zeros((10),dtype=np.float32)
lib = ctypes.cdll.LoadLibrary("./my_lib.so")
fun = lib.my_c_function
fun.restype = None
fun(ctypes.c_void_p(x.ctypes.data), ctypes.c_int(len(x)), ctypes.c_void_p(y.ctypes.data), ctypes.c_int(len(y)))
print y

*/

extern "C" void my_c_function(const void* x_void_ptr, const int x_size, void* y_void_ptr, const int y_size)
{
  const float* x = (float *)x_void_ptr;
  float* y = (float *)y_void_ptr;

  // do something, for example, assuming x is y_size by y_size matrix then the code below calculates the max of each column of x
  assert(y_size*y_size == x_size);
  for (int i=0; i<y_size; ++i) {
    const float* xx = x + i*y_size;
    y[i] = xx[0];
    for (int j=1; j<y_size; ++j) {
      y[i] = (xx[j] > y[i]) ? xx[j] : y[i];
    }
  }
  // end of example
  
}

