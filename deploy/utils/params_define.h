#ifndef Engine_params_define_h
#define Engine_params_define_h
// 输入、输出参数
#include <variant>
#include <string>
#include <map>
#include <vector> 
#include <opencv2/opencv.hpp>
namespace Engine
{
template <typename... Types>
struct JxVariant : public std::variant<Types...> {
    using std::variant<Types...>::variant;
    auto operator=(const char* rhs)
    {
        return std::variant<Types...>::operator=(std::string(rhs));
    }
};

using ValueType = JxVariant <
                  double,
                  float,
                  int,
                  bool,
                  std::string,
                  void*,
                  std::vector<double>,
                  cv::Size >;

using MapCalcParam = std::map<std::string, ValueType>;

template<typename T>
T GetParam(ValueType& val, const T& Def)
{
    auto pVal = std::get_if<T>(&val);
    if (pVal) {
        return *pVal;
    }
    //using U = std::decay_t<decltype(Def)>; // 类型退化，去掉类型中的const 以及 &
    //if constexpr (std::is_same_v<U, int>)
    //{
    //  auto pVal = std::get_if<int>(&val);
    //  if (pVal) return *pVal;
    //}
    return Def;
}
} // namespace Engine

#endif