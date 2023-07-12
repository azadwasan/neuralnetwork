#ifndef Commons_H
#define Commons_H
#include <utility>

namespace EasyNN {
	class Commons
	{
	public:
		//https://stackoverflow.com/questions/15208831/check-to-see-if-all-variable-are-equal-to-the-same-value-in-c
		template<typename T, typename U>
		static bool all_equal(T&& t, U&& u)
		{
			return (t == u);
		}

		template<typename T, typename U, typename... Ts>
		static bool all_equal(T&& t, U&& u, Ts&&... args)
		{
			return (t == u) && all_equal(u, std::forward<Ts>(args)...);
		}

	};
}


#endif