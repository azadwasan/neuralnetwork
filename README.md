A C++ implementation of neural networks from scratch. This is intended to for educational purposes and not to design a performant production quality library. However, initially results are very promising also in terms of performance and the accuracy.
We follow the following course from Andrew Ng and implement step by step all the concepts indtroducted during this course
https://www.coursera.org/specializations/machine-learning-introduction
Here are the core principles behind the design of the library
* Keep code as simple as possible
* Use C++ 20+ whenever possible
# Project Structure
The code has been created as a Visual Studio Solution, as it is extremely easy to setup and use the library within visual studio. However, this also means it is currently constrained to Windows only. Currently, we have the following projects in the solution
### EasyNN
This is the core neural network library containing all the necessary interfaces, algorithms and the framework. 
### EasyNNTest
Unit test project for EasyNN. Please note that we use the unit testing framework mostly for correctness (or even performance) testing and don't use it to implement the unit tests in the unit tests, e.g., checking boundary condition, ensure code coverage etc.
### EasyNNConsole
A console application that links with EasyNN (and other libraries in the solution) to play around. It doesn't serve more than quick testing. However, for appropriate use of EasyNN, EasyNNConsole should not be considered as a reference, rather EasyNNTest serves as an appropriate reference for apprropriate use of the library.
### EasyNNPythonDepictions
EasyNNTest does verify the correctness of the results, however, visualizing the results is often far more reassuring. Hence, EasyNNPythonDepictions, complements EasyNNtest project and often has graphical depictions for the results that have been verified by EasyNNTest.
### EasyNNPythonPlugin
Currently, our testing process involves generating various usecase data from Python libraries, such as 'make_regression'. However, this data is then manually copied to the tests in EasyNNTest, making the process cumbersome and tedious. Similarly, the reference solutions are generated using well-established Python libraries like Tensor, and again copied to the test code for verification. To streamline and simplify this process, I am working on the EasyNNPythonPlugin.

The main objective of the EasyNNPythonPlugin is to enable the execution of Python scripts directly from within C++. This means that we can now generate data by calling a Python script and create reference solutions by running another Python script. Finally, we can verify EasyNN results by comparing them with the reference results.

The progress on the EasyNNPythonPlugin has been promising so far, as we are already able to run Python scripts from within C++. However, there is still much more to be done. Specifically, we aim to implement functionality that allows us to call specific methods in the Python scripts, as well as efficiently pass and retrieve data to and from Python.
# Requirements
The following requirements are listed based on the development environment that I am currently using, though it is expected to work on a relatively diverse vareity of configurations
* Windows 10
* Visual Studio (Though, I am using Visual Studio Enterprise 2022, but I expect it to also work on Visual Studio Community edition https://visualstudio.microsoft.com/vs/community/)
* Visual Studio required workloads "Python Development" and "Desktop Development in C++"
* Python version 3.11.4 (This is the latest version at the time of development, however, I expect earlier 3.x.x versions to work too)
* Various Python packages. The requirements file is provided with the respective projects within the VS Solution.
# Setting up EasyNN development environment
Setting up EasyNN development environment is very simple and straight forward. 
1. Install Visual Studio (The community edition is available for free https://visualstudio.microsoft.com/vs/community/)
2. Choose the workloads "Python Development" and "Desktop Development in C++" while installing VS
3. "Python Development" workload will also install Python on the system, but in case it is desired, one can acquire the latest version from https://www.python.org/downloads/
4. Follow the instructions here https://learn.microsoft.com/en-us/visualstudio/version-control/git-clone-repository?view=vs-2022 to clone the EasyNN project (https://github.com/azadwasan/neuralnetwork)
6. Python requirements file is provided under both EasyNNPythonDepictions and EasyNNPythonPlugin. Install the required python packages as follows
   ```
   pip install -r requirements.txt
   ```
