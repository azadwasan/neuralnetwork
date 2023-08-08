# Introduction
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

In order to automate data and reference results fetch and testing I decided to extend EasyNN through an additional project, called EasyNNPythonPlugin, an embedded Python interpreter. This plugin enables the seamless integration of Python scripts within the C++ framework, combining the two worlds of C++ and Python, facilitating dynamic data generation and result verification.

![PythonEmbedding](/assets/img/EasyNNPythonEmbedding.png)


