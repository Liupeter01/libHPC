#pragma once
#ifndef _RETTYPE_HPP_
#define _RETTYPE_HPP_

#include <type_traits>

#if __cplusplus >= 201703L  // C++17 

template<typename Func, typename... Args>
using RetValue = std::invoke_result_t<Func, Args...>;

#elif __cplusplus >= 201103L  // C++11 ~ C++14 

template<typename Func, typename... Args>
using RetValue = typename std::result_of<Func(Args...)>::type;

#else

template<typename Func, typename... Args>
using RetValue = decltype(std::declval<Func>()(std::declval<Args>()...));

#endif

#endif // _RETTYPE_HPP_