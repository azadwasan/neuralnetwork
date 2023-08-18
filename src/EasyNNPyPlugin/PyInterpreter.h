#ifndef MYPYINTERPRETER_H
#define MYPYINTERPRETER_H
#include "NonCopyableNonAssignable.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <concepts>
#include <optional> 

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

namespace EasyNNPyPlugin {
    // The earlier implementation of PyInterpreter has a serious issue. If it is instantiated more than once,
    // the second time it fails to load python scripts, specially tesnsorflow. This happens because some python
    // libraries don't play so well if Py_Finalize() is called (which is what we did in the destructor) and
    // we try to Py_Initialize() again, these libraries, specifically numpy (on which tensorflow depends) doesn't
    // work anymore. Here are the details
    // https://stackoverflow.com/questions/7676314/py-initialize-py-finalize-not-working-twice-with-numpy
    // There are various ways we can address this issue.
    // 1. Implement PyInterpreter as a singleton 2. Call Py_Initialize() in a subclass are create a static instance
    // of this class in PyInterpreter 3. Don't call Py_Finalize() at all, as EasyNNPyPlugin is currently only used by
    // test project and is live only for a short duration until tests are complete. Moreover, calling Py_Initialize() 
    // multiple times is a noop if Py_Finalize() is not called.
    // Singleton feels liks a decent choice at the moment. Hence, we will use Mayer's singleton here and we will not call
    // Py_Finalize() for the reasons discussed above.

	class PyInterpreter final: private NonCopyableNonAssignable
	{
    private:
        PyInterpreter();
    public:
        static PyInterpreter& getInstance() {
            static PyInterpreter instance;
            return instance;
        }

        /**
         * @brief Convert C++ arguments to a Python tuple.
         *
         * This method takes a variable number of arguments (of any type) and converts them into a Python tuple.
         * The arguments are passed as a parameter pack, and the method uses recursive template specialization to create the tuple.
         *
         * @tparam Args Variadic template representing the types of arguments to be converted.
         * @param args The arguments to be converted into the Python tuple.
         *
         * @note The function can be used with any number of arguments of different types, and the order of the arguments
         * in the Python tuple will match the order in which they are passed to this function.
         *
         * @return PyObject* A pointer to the Python tuple representing the converted C++ arguments.
         *         The caller is responsible for managing the reference count of the returned PyObject.
         *         The returned PyObject is a new reference and should be decremented or released appropriately.
         *         If there is an error during conversion or memory allocation, NULL may be returned.
         *
         * @warning The function does not perform type checking or validation for the given arguments.
         *          It is the caller's responsibility to ensure that the arguments are compatible with Python types
         *          and that any memory management of the returned PyObject is handled correctly.
         *          Incorrect usage may lead to memory leaks or other undefined behavior.
         */        template <typename... Args>
        PyObject* convertArgumentsToPyTuple(Args... args) {
            const size_t numArgs = sizeof...(Args);

            PyObject* pTuple = PyTuple_New(numArgs);

            if (pTuple == nullptr) {
                PyErr_Print();
                // Handle error, e.g., throw an exception or return nullptr
                throw std::runtime_error("Failed to create a Python tuple from passed arguments.");
            }

            int index = 0;
            (void)fillTuple(pTuple, index, args...);
            return pTuple;
        }

		PyObject* executeMethod(const std::string& scriptName, const std::string& methodName, PyObject* args);

        /**
         * @brief Retrieve a matrix and a vector from a Python object.
         *
         * This method extracts a 2D matrix and a 1D vector from the given Python object,
         * which is expected to be a tuple containing the matrix as a list of lists and the vector as a list.
         *
         * @param matrix [out] A reference to a 2D std::vector<double> that will store the extracted matrix.
         * @param vector [out] A reference to a 1D std::vector<double> that will store the extracted vector.
         * @param args [in] A PyObject* representing the input Python object (a tuple containing the matrix and vector).
         *
         * @note The input Python object (args) is expected to be a tuple containing two elements:
         * - The first element is a list of lists, representing the 2D matrix.
         * - The second element is a list, representing the 1D vector.
         * Example: ([[4, 5, 6], [7, 8, 9]], [1, 2, 3]) corresponds to a matrix [[4, 5, 6], [7, 8, 9]] and a vector [1, 2, 3].
         *
         * @warning The method assumes that the input PyObject* is a valid tuple and the elements inside the tuple are lists.
         * It does not perform extensive error checking and should be used with care.
         * Morevoer, the order of the data to be retrieved is also a hard assumption.
         *
         * @return void
         */
        void retrieveMatrixAndVector(std::vector<std::vector<double>>& matrix, std::vector<double>& vector, PyObject* args);

        void extractMatrix(PyObject* pMatrixObj, std::vector<std::vector<double>>& matrix);
        void extractVector(PyObject* pVectorObj, std::vector<double>& vector);
    private:

        /**
         * @brief Helper function to fill a Python tuple with arguments.
         *
         * This template function is a helper used internally to fill a Python tuple with C++ arguments.
         * It uses recursion and template specialization to set each argument in the tuple.
         *
         * @tparam T The type of the first argument to be inserted into the tuple.
         * @tparam Args Variadic template representing the types of the remaining arguments.
         * @param pTuple [in] A pointer to the Python tuple to be filled.
         * @param index [in, out] An integer reference representing the current index within the tuple.
         *                        The function updates the index as each argument is added to the tuple.
         * @param arg [in] The first argument to be inserted into the tuple.
         * @param args [in] The remaining arguments to be inserted into the tuple.
         *
         * @note The function is used recursively to insert all the arguments into the tuple.
         *       The order of arguments in the Python tuple will match the order in which they are provided.
         *
         * @warning The function assumes that the caller passes a valid and appropriately sized Python tuple (pTuple),
         *          and that the index value is within the bounds of the tuple.
         *          Incorrect usage may lead to memory corruption or undefined behavior.
         *
         * @return void
         */

        template <typename T, typename... Args>
        inline void fillTuple(PyObject* pTuple, int& index, T arg, Args... args) {
            PyTuple_SetItem(pTuple, index++, ConvertArg(arg));
            fillTuple(pTuple, index, args...);
        }

        /**
         * @brief Helper function to fill a Python tuple with a single argument.
         *
         * This template function is a specialization of `fillTuple` used to insert a single argument into a Python tuple.
         * It sets the specified argument at the given index within the tuple.
         *
         * @tparam T The type of the argument to be inserted into the tuple.
         * @param pTuple [in] A pointer to the Python tuple to be filled.
         * @param index [in, out] An integer reference representing the index within the tuple where the argument will be inserted.
         *                        The function updates the index as the argument is added to the tuple.
         * @param arg [in] The argument to be inserted into the tuple.
         *
         * @note The function is used by the recursive `fillTuple` function to insert a single argument into the Python tuple.
         *
         * @warning The function assumes that the caller passes a valid and appropriately sized Python tuple (pTuple),
         *          and that the index value is within the bounds of the tuple.
         *          Incorrect usage may lead to memory corruption or undefined behavior.
         *
         * @return void
         */
        template <typename T>
        inline void fillTuple(PyObject* pTuple, int& index, T arg) {
            PyTuple_SetItem(pTuple, index++, ConvertArg(arg));
        }

        // Helper function to convert a single argument to a PyObject*
        template <typename T>
        PyObject* ConvertArg(T value) {
            if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    // Perform conversion for signed integral types
                    return PyLong_FromLongLong(static_cast<long long>(value));
                }
                else {
                    // Perform conversion for unsigned integral types
                    return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(value));
                }
            }
            else if constexpr (std::is_floating_point_v<T>) {
                return PyFloat_FromDouble(static_cast<double>(value));
            }
            else if constexpr (std::is_same_v<T, const char*>) {
                return PyUnicode_FromString(value);
            }
            else if constexpr (std::is_same_v<T, std::vector<double>>) {
                PyObject* pList = PyList_New(value.size());
                if (pList == nullptr) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to create a Python list from std::vector<std::vector<double>>.");
                }
                for (size_t i = 0; i < value.size(); ++i) {
                    PyList_SetItem(pList, i, PyFloat_FromDouble(value[i]));
                }
                return pList;
            }
            else if constexpr (std::is_same_v<T, std::vector<std::vector<double>>>) {
                PyObject* pList = PyList_New(value.size());
                if (pList == nullptr) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to create a Python list from std::vector<std::vector<double>>.");
                }
                for (size_t i = 0; i < value.size(); ++i) {
                    PyObject* pInnerList = PyList_New(value[i].size());
                    if (pInnerList == nullptr) {
                        PyErr_Print();
                        throw std::runtime_error("Failed to create a Python list from std::vector<std::vector<double>>.");
                    }
                    for (size_t j = 0; j < value[i].size(); ++j) {
                        PyList_SetItem(pInnerList, j, PyFloat_FromDouble(value[i][j]));
                    }
                    PyList_SetItem(pList, i, pInnerList);
                }
                return pList;
            }
            else if constexpr (std::is_same_v < T, std::optional<std::vector<double>>>) {
                if (value == std::nullopt) {
                    return Py_None;
                }
                else {
                    PyObject* pList = PyList_New(value->size());
                    if (pList == nullptr) {
                        PyErr_Print();
                        throw std::runtime_error("Failed to create a Python list from std::vector<std::vector<double>>.");
                    }
                    for (size_t i = 0; i < value->size(); ++i) {
                        PyList_SetItem(pList, i, PyFloat_FromDouble(value.value()[i]));
                    }
                    return pList;
                }
            }
            else if constexpr (std::is_same_v < T, std::optional<size_t>>) {
                if (value == std::nullopt) {
                    return Py_None;
                }
                else {
                    return PyLong_FromLongLong(static_cast<size_t>(*value));
                }
            }
            else {
                throw std::runtime_error("Unsupported argument type.");
            }
        }
	private:
	};
}
#endif