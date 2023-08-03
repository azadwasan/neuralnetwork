#ifndef NONCOPYABLENONASSIGNABLE_H
#define NONCOPYABLENONASSIGNABLE_H

namespace EasyNNPyPlugin {
	class NonCopyableNonAssignable {
	protected:
		NonCopyableNonAssignable() = default;
		// Declare the copy constructor and copy assignment operator as deleted
		NonCopyableNonAssignable(const NonCopyableNonAssignable&) = delete;
		NonCopyableNonAssignable& operator=(const NonCopyableNonAssignable&) = delete;
	};

}

#endif