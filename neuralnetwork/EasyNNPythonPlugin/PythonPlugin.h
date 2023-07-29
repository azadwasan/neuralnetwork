#pragma once
#include <iostream>
//https://stackoverflow.com/questions/17028576/using-python-3-3-in-c-python33-d-lib-not-found
#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

void aFunction() {
	Py_Initialize();
	PyRun_SimpleString("print('Hello from Python!')");
	Py_Finalize();

	std::cout << "This is a python plugin " << std::endl;
}