#pragma once
#include <iostream>
#include <string>
#include <vector>

//https://stackoverflow.com/questions/17028576/using-python-3-3-in-c-python33-d-lib-not-found
#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

void aFunction() {
    std::string scriptName{ "TestScript"};
    std::string methodName {"MyTestMethod"};
    std::wstring scriptDirectory = L"../EasyNNPyScripts";

    // Set the working directory to the location containing the Python script file
    // Replace "your_working_directory" with the actual path
#ifdef _WIN32
    SetCurrentDirectory(scriptDirectory.c_str());
#else
    chdir(scriptDirectory.c_str());
#endif
    // Initialize Python interpreter
    Py_Initialize();

    // Append the script directory to the Python sys.path list
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

   //// PyObject* sysPath = PySys_GetObject("path");
   // if (sysPath != nullptr && PyList_Check(sysPath)) {
   //     Py_ssize_t numPaths = PyList_Size(sysPath);
   //     for (Py_ssize_t i = 0; i < numPaths; ++i) {
   //         PyObject* pathItem = PyList_GetItem(sysPath, i);
   //         if (pathItem != nullptr && PyUnicode_Check(pathItem)) {
   //             const wchar_t* pathStr = PyUnicode_AsWideCharString(pathItem, nullptr);
   //             wprintf(L"sys.path[%zd]: %s\n", i, pathStr);
   //             PyMem_Free((void*)pathStr);
   //         }
   //     }
   // }

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

    // Create a sample vector
    std::vector<int> inputData = { 1, 2, 3, 4, 5 };

    // Convert the vector to a Python list
    PyObject* pList = PyList_New(inputData.size());
    for (size_t i = 0; i < inputData.size(); ++i) {
        PyObject* pValue = PyLong_FromLong(inputData[i]);
        PyList_SetItem(pList, i, pValue);
    }

    // Call the method with the Python list as an argument
    PyObject* pArgs = PyTuple_Pack(1, pList);
    PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

    // Check if the call was successful
    if (pResult != nullptr) {
        Py_ssize_t resultSize = PyList_Size(pResult);
        std::vector<int> resultData;

        for (Py_ssize_t i = 0; i < resultSize; ++i) {
            PyObject* pItem = PyList_GetItem(pResult, i);
            if (pItem != nullptr && PyLong_Check(pItem)) {
                int value = PyLong_AsLong(pItem);
                resultData.push_back(value);
            }
            else {
                throw std::runtime_error("Failed to retrieve result element.");
            }
        }

        printf("Result from Python:");
        for (int value : resultData) {
            printf(" %d", value);
        }
        printf("\n");

        Py_DECREF(pResult);
    }
    else {
        PyErr_Print(); // Print any Python exceptions
        throw std::runtime_error("Python method call failed.");
    }

    // Followng code snippets are for passing in vector<vector<>> and a scaler.

    //// Create a sample vector<std::vector<int>> representing a matrix
    //std::vector<std::vector<int>> inputMatrix = {
    //    {1, 2, 3},
    //    {4, 5, 6},
    //    {7, 8, 9}
    //};

    //// Convert the matrix to a Python list of lists
    //PyObject* pMatrix = PyList_New(inputMatrix.size());
    //for (size_t i = 0; i < inputMatrix.size(); ++i) {
    //    PyObject* pRow = PyList_New(inputMatrix[i].size());
    //    for (size_t j = 0; j < inputMatrix[i].size(); ++j) {
    //        PyObject* pValue = PyLong_FromLong(inputMatrix[i][j]);
    //        PyList_SetItem(pRow, j, pValue);
    //    }
    //    PyList_SetItem(pMatrix, i, pRow);
    //}

    //// Create a scalar value (integer)
    //int scalarValue = 10;
    //PyObject* pScalar = PyLong_FromLong(scalarValue);

    //// Call the method with the Python list of lists (matrix) and the scalar as arguments
    //PyObject* pArgs = PyTuple_Pack(2, pMatrix, pScalar);
    //PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

    //// Check if the call was successful
    //if (pResult != nullptr && PyList_Check(pResult)) {
    //    Py_ssize_t numRows = PyList_Size(pResult);
    //    std::vector<std::vector<int>> resultMatrix;

    //    for (Py_ssize_t i = 0; i < numRows; ++i) {
    //        PyObject* pRow = PyList_GetItem(pResult, i);
    //        if (pRow != nullptr && PyList_Check(pRow)) {
    //            Py_ssize_t numCols = PyList_Size(pRow);
    //            std::vector<int> rowValues;

    //            for (Py_ssize_t j = 0; j < numCols; ++j) {
    //                PyObject* pItem = PyList_GetItem(pRow, j);
    //                if (pItem != nullptr && PyLong_Check(pItem)) {
    //                    int value = PyLong_AsLong(pItem);
    //                    rowValues.push_back(value);
    //                }
    //                else {
    //                    throw std::runtime_error("Failed to retrieve matrix element.");
    //                }
    //            }

    //            resultMatrix.push_back(rowValues);
    //        }
    //        else {
    //            throw std::runtime_error("Failed to retrieve matrix row.");
    //        }
    //    }

    //    printf("Resulting matrix:\n");
    //    for (const auto& row : resultMatrix) {
    //        for (int value : row) {
    //            printf("%d ", value);
    //        }
    //        printf("\n");
    //    }

    //    Py_DECREF(pResult);
    //}
    //else {
    //    PyErr_Print(); // Print any Python exceptions
    //    throw std::runtime_error("Python method call failed.");
    //}





    // Cleanup
    Py_DECREF(pArgs);
    Py_DECREF(pList);
    Py_DECREF(pMethod);
    Py_DECREF(pModule);

    // Finalize Python interpreter
    Py_Finalize();
}