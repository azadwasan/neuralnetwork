This project contains various Python scripts for data generation and depictions of the results used by the EasyNNTest project.
There are often one to one corresponding scripts to the EasyNNTest files, e.g., LinearHypothesisText.cpp has a cooresponding script LinearHypothesis.py.
In addition, this project contains scripts to
* generate data for the tests in EasyNNTest project
* generate reference solutions from Tensor Flow algorithms to compare the resutls of EasyNN implementations in EasyNNTest project

The plan is to develop EasyNNPythonPlugin and use it to directly run the scripts from EasyNNPySciprts to generate data, reference results from tests in EasyNNTest project and possibily plot results etc.
 
