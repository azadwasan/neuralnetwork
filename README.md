Inspired by Andrew Ng's renowned Machine Learning course (https://www.coursera.org/specializations/machine-learning-introduction), a C++ library to implement Neural Networks from scratch.
With a focus on accessibility, EasyNN is designed to complement Andrew's course, making the implementation as learner-friendly as possible. My primary goal is to ensure that newcomers to machine learning can easily grasp and utilize the library to dive into the world of neural networks.
While keeping ease-of-understanding in mind, I've also aimed to make EasyNN both extensible and highly performant.
I'm already excited to share that some early milestones have been achieved! EasyNN now boasts a Linear Hypothesis and efficient Gradient Descent implementation, both of which have undergone rigorous testing.
In my initial performance analysis, I compared EasyNN against TensorFlow, and the results are looking incredibly promising!

# Current State of the Project
EasyNN (Core neural network library): Linear Hypothesis, Mean Square Error Cost function, Gradient Descent The progress so far corresponds to Course 1, Week 2 of Machine Learning Specialization (https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction)
EasyNNPythonScripts: Full Python embedded (that we need for now). We can call a arbitrary method from any python module from our C++ code. This is very useful for EasyNNTest project to retrieve data and reference results from Tesnsorflow.
Next steps:
* Update EasyNNTest to use dynamic data and reference results from Python.
* Move back to EasyNN and continue with the core library development.

# Project Structure
The code has been created as a Visual Studio Solution, as it is extremely easy to setup and use the library within visual studio. However, this also means it is currently constrained to Windows only. Currently, we have the following projects in the solution
## EasyNN
This is the core neural network library containing all the necessary interfaces, algorithms and the framework.
### Current State
EasyNN core library currently implements the following
* Interface for Hypothesis and Cost function
* Linear Hypothesis
* Mean Square Error Cost function
* Gradient Descent
The progress so far corresponds to Course 1, Week 2 of Machine Learning Specialization (https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction)
### Next Steps
Next step is to implement logistic regression, which will correspond to Course 1, Week 2 of the above.
### EasyNNTest
Unit test project for EasyNN. Please note that we use the unit testing framework mostly for correctness (or even performance) testing and don't use it to implement the unit tests in the unit tests, e.g., checking boundary condition, ensure code coverage etc.
### Current State
EasyNNtest is in sync with EasyNN core library, i.e., the necessary tests that I needed to implement to test the correctness of EasyNN are implemented.
### Key takeaways
Testing EasyNN implementation required finding appropriate refrence data and the corresponding solution, which is becoming time consuming as I am progressing further. Hence, I came up with the idea of EasyNNPythonScripts project to generate the reference data and solutions. However, it was still cumbersome, hence, I came up with an other idea of automating it all using EasyNNPythonPlugin. Read description of the corresponding projects for further details.
## EasyNNConsole
A console application that links with EasyNN (and other libraries in the solution) to play around. It doesn't serve more than quick testing. However, for appropriate use of EasyNN, EasyNNConsole should not be considered as a reference, rather EasyNNTest serves as an appropriate reference for apprropriate use of the library.
## EasyNNPythonScripts
EasyNNTest does verify the correctness of the results, however, visualizing the results is often far more reassuring. Hence, EasyNNPythonScripts, complements EasyNNtest project and often has graphical depictions for the results that have been verified by EasyNNTest.
## EasyNNPythonPlugin
Currently, our testing process involves generating various usecase data from Python libraries, such as 'make_regression'. However, this data is then manually copied to the tests in EasyNNTest, making the process cumbersome and tedious. Similarly, the reference solutions are generated using well-established Python libraries like Tensor, and again copied to the test code for verification. To streamline and simplify this process, I am working on the EasyNNPythonPlugin.

The main objective of the EasyNNPythonPlugin is to enable the execution of Python scripts directly from within C++. This means that we can now generate data by calling a Python script and create reference solutions by running another Python script. Finally, we can verify EasyNN results by comparing them with the reference results.

### Current state
The progress on the EasyNNPythonPlugin has been promising so far, as we are already able to run Python scripts from within C++. However, there is still much more to be done. 
### Next steps
Call specific methods in the Python scripts, as well as efficiently pass and retrieve data to and from Python.

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
5. Python requirements file is provided under both EasyNNPythonScripts and EasyNNPythonPlugin. Install the required python packages as follows
   ```
   pip install -r requirements.txt
   ```
# Using EasyNN*
Once the development environment has been setup, the natural next step is to try to build the project. EasyNN project contains the code for the core algorithms related to neural network. However, as it is written as a library, hence it needs to be linked to either an executable or a test. A skeleton console app has been provided witht he solution that is already linked to the EasyNN core library and can be used to play with EasyNN library. However, the best starting points is to look at EasyNNtest project and go through the test code and  try to run the tests through test explorer. The test code is generally very well documented and demonstrates step by step how to use various EasyNN core library functionalities.
