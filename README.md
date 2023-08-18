Inspired by Andrew Ng's renowned Machine Learning course (https://www.coursera.org/specializations/machine-learning-introduction), a C++ library to implement Neural Networks from scratch.
With a focus on accessibility, EasyNN is designed to complement Andrew's course, making the implementation as learner-friendly as possible. My primary goal is to ensure that newcomers to machine learning can easily grasp and utilize the library to dive into the world of neural networks.
While keeping ease-of-understanding in mind, I've also aimed to make EasyNN both extensible and highly performant.
I'm already excited to share that some early milestones have been achieved! EasyNN now boasts a Linear Hypothesis and efficient Gradient Descent implementation, both of which have undergone rigorous testing.
In my initial performance analysis, I compared EasyNN against TensorFlow, and the results are looking incredibly promising!

# Documentation

The project documentation can be found here
https://azadwasan.github.io/neuralnetwork/

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
