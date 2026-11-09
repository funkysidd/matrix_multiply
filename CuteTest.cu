#include "cute/container/tuple.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "cute/stride.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include <iostream>

using namespace cute;

int main()
{
    auto layout = make_layout(make_shape(_4{}, _4{}));

    std::cout << layout << std::endl;

    return 0;
}
