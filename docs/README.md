![image](https://github.com/azadwasan/neuralnetwork/assets/6748027/23bee843-55bf-44ad-82d3-c5f033de9c57)# Introduction
The landscape of machine learning has witnessed remarkable strides in recent years, with neural networks emerging as a pivotal force behind cutting-edge advancements. However, this domain's intricacies often present a steep learning curve, particularly for those new to the field. To bridge this gap, we proudly introduce EasyNN, a beginner-friendly C++ library meticulously designed to facilitate the implementation of neural networks from the ground up. With a strong focus on accessibility and high performance, EasyNN aims to guide newcomers through the world of neural networks.

# The Genesis of EasyNN
Inspired by the transformative Machine Learning course offered by Andrew Ng, the idea for EasyNN took root. This renowned course successfully introduced neural networks to a wide audience, equipping learners with the ability to utilize existing libraries like TensorFlow for practical applications. However, while learners gained insights into using neural networks, there was a crucial gap in understanding their fundamental implementation. The missing piece lay in bridging the theoretical concepts with their practical realization, beyond the scope of utilizing pre-existing frameworks. EasyNN's genesis is rooted in this realization - to create a supportive environment for learners to engage in experimentation, innovation, and the process of unraveling the intricacies underlying neural networks.

# A Solution to Testing EasyNN
As the development of EasyNN progressed, a crucial challenge emerged: testing the library effectively. The process of sourcing reference data and establishing benchmark solutions for various algorithms proved to be time-intensive. While Python boasts robust capabilities in data generation and neural network frameworks, the challenge remained in assessing EasyNN's performance against established solutions like TensorFlow. Despite using Python libraries for dynamic reference data and results, the process was still laborious, involving multiple manual steps:

* Generating the data (Python)
* Transforming the data and generating results from EasyNN (C++)
* Generating reference results from Tesnsorflow (Python)
* Consolidating the two results and writing the tests (C++)

## Python Embedding

In order to automate the retrieval of data and reference results, EasyNN was extended through an adjunct  project, called EasyNNPythonPlugin, an embedded Python interpreter. This plugin enables the seamless integration of Python scripts within the C++ framework, combining the two worlds of C++ and Python, facilitating dynamic data generation and result verification. The current architecture of EasyNN solution looks as follows:

![PythonEmbedding](/assets/img/EasyNNPythonEmbedding.png)

EasyNNPyScripts are the python scripts for various operations, e.g., generating regression data, gradient descent optimization using tesnsorflow. EasyNNPyPlugin is the embedded python interpreter and is designed as a library to link against. EasyNN is the core library to implement various neural networks related algorithms. EasyNNTest is a MSTest C++ test project that links against both EasyNN and EasyNNPlugin. It runs various python scripts from EasyNNPyScripts  through EasyPyPlugin to fetch the generated data and refrences results from tensorflow and feeds the same data to EasyNN core library and compares the two results. It also uses EasyNNPlugin to run various python scripts to plot the results from both EasyNN and Tensorflow.

# Prelimnary Results

Linear Hypothesis, Mean Square Error Cost function and Gradient Descent have been implemented in EasyNN. Using the powerful testing infrastructure to test EasyNN against Tensoflow, prelimnary results have been acquired. Though, no extensive benchmarking has been done, but the intial results are extremely promising, so much so that EasyNN gradient and Tensorflow gradient descent optimization are virtually indistinguishable. However, in rare cases if we have very few samples, as low as 5, it appears the optimization performed by EasyNN and Tensorflow appear to diverge. The following figure depicts such rare case (achieved after multiple try runs), where we have two features and only 5 samples and a linear hypothesis is approximated using gradient descent optimization:

![PythonEmbedding](/assets/img/EasyNNvsTensorFlowGD.png)

The green and blue planes represent the linear approximation by EasyNN and tesnsorflow using gradient descent optimization respectively. However, if we would rotate the graph, it becomes clear that both achieve similar level of approximation, as was also clear from the MSE when checked.

The following graph depits the approximation computed by both EasyNN and Tensorflow using graident descent for 500 data points with very high dispersion. The results from both are so close that either only the blue or the green plane could be seen by rotating.

![PythonEmbedding](/assets/img/EasyNNvsTensorFlowGD2.png)

Though EasyNN results are being compared with Tensorflow in order to verify the accuracy of the implementation, however, a positive surprise of such comparison was a dramatic different in performance. EasyNN performs GD optimization for 10 features and 100 samples in 9.18 miliseconds, whereas Tensorflow require 2500 miliseconds. Though, this results must be taken with a grain of salt as we are comparing EasyNN release version performance with Tensorflow python unoptimized version.  

# Project Structure
The code has been created as a Visual Studio Solution, as it is extremely easy to setup and use EasyNN library within visual studio. However, this also means it is currently constrained to Windows only. The visual studio solution consists of the following projects:
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
Testing EasyNN implementation required finding appropriate refrence data and the corresponding solution, which is becoming time consuming as I am progressing further. Hence, I came up with the idea of EasyNNPythonScripts project to generate the reference data and solutions. However, it was still cumbersome to manually copy the data, modify it to C++ syntex etc., hence, I planned to automate it all using EasyNNPythonPlugin. Read description of the corresponding projects for further details.
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

# Current State of the Project

**EasyNN** (Core neural network library): Linear Hypothesis, Mean Square Error Cost function, Gradient Descent The progress so far corresponds to Course 1, Week 2 of Machine Learning Specialization (https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction)

**EasyNNPythonScripts** : Full Python embedded (that we need for now). We can call a arbitrary method from any python module from our C++ code. This is very useful for EasyNNTest project to retrieve data and reference results from Tesnsorflow.
Next steps:
* Update EasyNNTest to use dynamic data and reference results from Python.
* Move back to EasyNN and continue with the core library development.
