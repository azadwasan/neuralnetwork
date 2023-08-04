#ifndef COMMON_H
#define COMMON_H
#include <memory>

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

namespace EasyNNPyPlugin {
    // Define a custom deleter that calls Py_DECREF
    struct PyObjectDeleter {
        void operator()(PyObject* obj) const {
            Py_XDECREF(obj);//Py_XDECREF in compariosn to Py_DECREF() performs a null check and is a bit slower!
        }
    };
    using easyNN_unique_ptr = std::unique_ptr<PyObject, PyObjectDeleter>;
}
#endif
