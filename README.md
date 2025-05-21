# micrograd-cpp

A minimal C++ implementation of [micrograd](https://github.com/karpathy/micrograd) — an automatic differentiation engine for scalar-valued expressions. This project supports forward computation and reverse-mode autodiff (i.e. backpropagation).

## Features

- Core `Value` class for scalar values with gradient tracking
- Operator overloading for `+`, `*`, etc.
- Support for `tanh()` activation
- Reverse-mode automatic differentiation via `backward()`
- Easy to extend and experiment with

## Example

```cpp
auto x1 = std::make_shared<Value>(2.0, {}, "", "x1");
auto w1 = std::make_shared<Value>(-3.0, {}, "", "w1");
auto b  = std::make_shared<Value>(6.88, {}, "", "b");

auto out = (x1 * w1 + b)->tanh();
out->grad_ = 1.0;
out->backward();

std::cout << "Gradient w.r.t x1: " << x1->grad_ << std::endl;
