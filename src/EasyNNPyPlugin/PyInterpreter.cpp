#include "pch.h"
#include "PyInterpreter.h"
#include <stdexcept>

// Required to set the path to scripts directory, otherwise interpreter wouldn't be able to find it
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

using namespace EasyNNPyPlugin;



const std::wstring PyInterpreter::scriptDirectory = L"../EasyNNPyScripts";

PyInterpreter::PyInterpreter() {

    // Set the working directory to the location containing the Python script file
#ifdef _WIN32
    SetCurrentDirectory(scriptDirectory.c_str());
#else
    chdir(scriptDirectory.c_str());
#endif
    // Initialize Python interpreter
    Py_Initialize();

    // Append the script directory to the Python sys.path list, 
    // otherwise if the script calls another script in the same directory it will fail.
    PyObject* sysPath = PySys_GetObject("path");
    if (sysPath != nullptr && PyList_Check(sysPath)) {
        PyObject* directoryPath = PyUnicode_FromWideChar(scriptDirectory.c_str(), scriptDirectory.size());
        if (directoryPath != nullptr) {
            PyList_Append(sysPath, directoryPath);
            Py_DECREF(directoryPath); // Release reference to the directory path object
        }
        else {
            throw std::runtime_error("Failed to create directory path object.");
        }
    }
    else {
        throw std::runtime_error("sys.path is not a valid list object.");
    }
}
PyInterpreter::~PyInterpreter() {
    // Finalize Python interpreter
    Py_Finalize();
}

PyObject* PyInterpreter::executeMethod(const std::string& scriptName, const std::string& methodName, PyObject* args) {
    // Load the Python script module
    PyObject* pModule = PyImport_ImportModule(scriptName.c_str());
    if (pModule == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to load the Python script module.");
    }

    // Get a reference to the method within the module
    PyObject* pMethod = PyObject_GetAttrString(pModule, methodName.c_str());
    if (pMethod == nullptr || !PyCallable_Check(pMethod)) {
        PyErr_Print();
        throw std::runtime_error("Failed to get the function.");
    }

    //// Call the method with the Python list as an argument
    PyObject* pResult = PyObject_CallObject(pMethod, args);

    Py_DECREF(pMethod);
    Py_DECREF(pModule);

    return pResult;
}

void PyInterpreter::retrieveMatrixAndVector(std::vector<std::vector<double>>& matrix, std::vector<double>& vector, PyObject* pResult) {
    // Extract the first item from the tuple (matrix)
    PyObject* pMatrixObj = PyTuple_GetItem(pResult, 0);
    if (pMatrixObj != nullptr && PyList_Check(pMatrixObj)) {
        extractMatrix(pMatrixObj, matrix);
    }
    else {
        PyErr_Print(); // Print any Python exceptions
        throw std::runtime_error("Failed to retrieve matrix.");
    }

    // Extract the second item from the tuple (vector)
    PyObject* pVectorObj = PyTuple_GetItem(pResult, 1);
    if (pVectorObj != nullptr && PyList_Check(pVectorObj)) {
        extractVector(pVectorObj, vector);
    }
    else {
        throw std::runtime_error("Failed to retrieve vector.");
    }
}

void PyInterpreter::extractMatrix(PyObject* pMatrixObj, std::vector<std::vector<double>>& matrix) {
    Py_ssize_t numRows = PyList_Size(pMatrixObj);

    for (Py_ssize_t i = 0; i < numRows; ++i) {
        PyObject* pRow = PyList_GetItem(pMatrixObj, i);
        if (pRow != nullptr && PyList_Check(pRow)) {
            std::vector<double> rowValues;
            extractVector(pRow, rowValues);
            matrix.push_back(rowValues);
        }
        else {
            throw std::runtime_error("Failed to retrieve matrix row.");
        }
    }
}

void PyInterpreter::extractVector(PyObject* pVectorObj, std::vector<double>& vector) {
    Py_ssize_t numElements = PyList_Size(pVectorObj);

    for (Py_ssize_t i = 0; i < numElements; ++i) {
        PyObject* pItem = PyList_GetItem(pVectorObj, i);
        if (pItem != nullptr && PyFloat_Check(pItem)) {
            double value = PyFloat_AsDouble(pItem);
            vector.push_back(value);
        }
        else {
            throw std::runtime_error("Failed to retrieve vector element.");
        }
    }
}
