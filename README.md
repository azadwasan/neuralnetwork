A C++ implementation of neural networks from scratch. This is intended to for educational purposes and not to design a performant production quality library. However, initially results are very promising also in terms of performance and the accuracy.
We follow the following course from Andrew Ng and implement step by step all the concepts indtroducted during this course
https://www.coursera.org/specializations/machine-learning-introduction
Here are the core principles behind the design of the library
* Keep code as simple as possible
* Use C++ 20+ whenever possible
# Project Structure
The code has been created as a Visual Studio Solution, as it is extremely easy to setup and use the library within visual studio. However, this also means it is currently constrained to Windows only. Currently, we have the following projects in the solution
## EasyNN
This is the core neural network library containing all the necessary interfaces, algorithms and the framework. 
## EasyNNTest
Unit test project for EasyNN. Please note that we use the unit testing framework mostly for correctness (or even performance) testing and don't use it to implement the unit tests in the unit tests, e.g., checking boundary condition, ensure code coverage etc.
## EasyNNConsole
A console application that links with EasyNN (and other libraries in the solution) to play around. It doesn't serve more than quick testing. However, for appropriate use of EasyNN, EasyNNConsole should not be considered as a reference, rather EasyNNTest serves as an appropriate reference for apprropriate use of the library.
## EasyNNPythonDepictions
EasyNNTest does verify the correctness of the results, however, visualizing the results is often far more reassuring. Hence, EasyNNPythonDepictions, complements EasyNNtest project and often has graphical depictions for the results that have been verified by EasyNNTest.
## EasyNNPythonPlugin
Currently, various usecase data are generated from Python libraries, e.g., make_regression, that are then manually copied to the tests in EasyNNTest. Once again, the reference solution are generated for the data using well established python libraries like Tensor, which are again copied to the test code and then verified. However, it is very cumbersome and tedious process. 
Hence, the idea of EasyNNPythonPlugin, which allows the python scripts to be executed directly from within C++, e.g., by a test that generates the data directly by calling a python script and then generate reference solution by running another python script and finally verify EasyNN results by comparing them with the reference results.
This work is in progress. So far, we can already run the python scripts, but a lot more is to be done yet, e.g., calling specific methods, passing and retrieving data to and from Python etc.
