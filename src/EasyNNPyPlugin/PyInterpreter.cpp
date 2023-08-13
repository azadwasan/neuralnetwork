#include "pch.h"
#include "PyInterpreter.h"
#include "Common.h"

#include <stdexcept>
#include <cstdlib>

// Required to set the path to scripts directory, otherwise interpreter wouldn't be able to find it
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
using namespace EasyNNPyPlugin;

PyInterpreter::PyInterpreter() {
    // Before this code we had the relative path of the scripts folder specified as a static member variable. 
    // However, it doesn't work if we use this library with a MS Test and test explorer, as the path of the test 
    // explorer is not the same as this solution, hence the relative path become invalid. Hence, I had to resort 
    // to the following solution.
    // However, this is a very bad idea and must not be used in production software. I am using it here only because
    // this code is only supposed to be used within a development evnrionment of Visual Studio and not in a production
    // environment, hence it should work. In the case of production, we will have to use environmen variables to set
    // the appropriate paths to the scripts folder.

    // Construct the path to the EasyNNPyScripts directory relative to the current source file
    std::string currentFilePath = __FILE__;
    std::wstring currentFilePathW;
    currentFilePathW.resize(currentFilePath.size());
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, &currentFilePathW[0], currentFilePathW.size(), currentFilePath.c_str(), _TRUNCATE);
    std::wstring scriptDirectory = currentFilePathW.substr(0, currentFilePathW.find_last_of(L"/\\") + 1) + L"../EasyNNPyScripts";

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
        easyNN_unique_ptr directoryPath{ PyUnicode_FromWideChar(scriptDirectory.c_str(), scriptDirectory.size()) };
        if (directoryPath != nullptr) {
            PyList_Append(sysPath, directoryPath.get());
        }
        else {
            throw std::runtime_error("Failed to create directory path object.");
        }
    }
    else {
        throw std::runtime_error("sys.path is not a valid list object.");
    }
}

PyObject* PyInterpreter::executeMethod(const std::string& scriptName, const std::string& methodName, PyObject* args) {
    easyNN_unique_ptr pModule{ PyImport_ImportModule(scriptName.c_str()) };

    if (pModule == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to load the Python script module.");
    }

    // Get a reference to the method within the module
    easyNN_unique_ptr pMethod{ PyObject_GetAttrString(pModule.get(), methodName.c_str())};
    if (pMethod == nullptr || !PyCallable_Check(pMethod.get())) {
        PyErr_Print();
        throw std::runtime_error("Failed to get the function.");
    }

    //// Call the method with the Python list as an argument
    return PyObject_CallObject(pMethod.get(), args);
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

// We use the following method both for extracting the vectors and also each line
// of the matrix (which are actually vectors).
void PyInterpreter::extractVector(PyObject* pResult, std::vector<double>& vector) {
    PyObject* pVectorObj = pResult;
    if (PyTuple_Check(pResult)) {
        pVectorObj = PyTuple_GetItem(pResult, 0);
    }
    Py_ssize_t numElements = PyList_Size(pVectorObj);

    for (Py_ssize_t i = 0; i < numElements; ++i) {
        PyObject* pItem = PyList_GetItem(pVectorObj, i);
        if (pItem != nullptr) {
            if (PyFloat_Check(pItem)) {
                vector.push_back(PyFloat_AsDouble(pItem));
            }
            else if (PyLong_Check(pItem)) {
                vector.push_back(PyLong_AsDouble(pItem));
            }
            else {
                throw std::runtime_error("Type of the python list element unrecognized.");
            }
        }
        else {
            throw std::runtime_error("Python list item null. Failed to convert the list to vector.");
        }
    }
}
