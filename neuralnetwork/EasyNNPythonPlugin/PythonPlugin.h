#pragma once
#include <iostream>
#include <string>
//https://stackoverflow.com/questions/17028576/using-python-3-3-in-c-python33-d-lib-not-found
#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

void aFunction() {
	std::string fileName{ "..\\EasyNNPyScripts\\LinearHypothesis.py" };

// Replace this with the actual path to the directory containing the script you want to run
	std::wstring scriptDirectory = L"..\\EasyNNPyScripts";

	// Initialize Python interpreter
	Py_Initialize();

    // Append the script directory to the Python sys.path list
    PyObject* sysPath = PySys_GetObject("path");
    if (sysPath != nullptr && PyList_Check(sysPath)) {
        PyObject* directoryPath = PyUnicode_FromWideChar(scriptDirectory.c_str(), scriptDirectory.size());
        if (directoryPath != nullptr) {
            auto result = PyList_Append(sysPath, directoryPath);
            Py_DECREF(directoryPath); // Release reference to the directory path object
            if (result == -1) {
                throw std::runtime_error("Failed to append path to sys.path.");
            }
        }
        else {
            throw std::runtime_error("Failed to create directory path object.");
        }
    }
    else {
        throw std::runtime_error("sys.path is not a valid list object.");
    }

    //Simple file open will not work, hence we have to do as described in the stacoverflow post.
    //https://stackoverflow.com/questions/65444044/embedding-python-in-c-script-pyrun-simplefile-not-working-as-expected
    PyObject* obj = Py_BuildValue("s", fileName.c_str());
    FILE* file = _Py_fopen_obj(obj, "r+");
    if (file != NULL) {
        PyRun_SimpleFile(file, fileName.c_str());
    }

	Py_Finalize();
}