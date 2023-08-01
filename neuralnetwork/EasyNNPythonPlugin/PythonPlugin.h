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
	//std::string fileName{ "..\\EasyNNPythonDepictions\\TestScript.py" };
	std::string fileName{ "..\\EasyNNPythonDepictions\\LinearHypothesis.py" };

	Py_Initialize();

	//Simple file open will not work, hence we have to do as described in the stacoverflow post.
	//https://stackoverflow.com/questions/65444044/embedding-python-in-c-script-pyrun-simplefile-not-working-as-expected
	PyObject* obj = Py_BuildValue("s", fileName.c_str());
	FILE* file = _Py_fopen_obj(obj, "r+");
	if (file != NULL) {
		PyRun_SimpleFile(file, fileName.c_str());
	}

	Py_Finalize();
}