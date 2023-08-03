#ifndef COMMON_H
#define COMMON_H

#include <string>
#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace EasyNNPyPlugin {
    class Common final
    {
    public:
    private:
        Common() = default;

        Common(const Common&) = delete;
        Common& operator=(const Common&) = delete;
    };


    //void myMain() {
    //    // Prepare the arguments
    //    int arg1 = 10;
    //    int arg2 = 20;
    //    double arg3 = 3.14;
    //    const char* arg4 = "Hello";

    //    // Convert the arguments to a PyObject* tuple using the Common class
    //    PyObject* pArgs = Common::convertArgumentsToPyTuple(arg1, arg2, arg3, arg4);

    //    // Call the method with the Python arguments
    //    //PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

    //    // ... (Handle the result, cleanup, and finalize Python interpreter)
    //}
}
#endif
