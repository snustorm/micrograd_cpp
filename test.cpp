#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <functional>

#include "ops.h"
#include "value.h"


int main() {
    std::cout << "Micrograd!" << std::endl;

    auto x = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "a");

    auto a = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "a");
    auto b = std::make_shared<Value>(-3.0, std::vector<std::shared_ptr<Value>>{}, "", "b");
    auto c = std::make_shared<Value>(10.0, std::vector<std::shared_ptr<Value>>{}, "", "c");

    // a.grad_ = 6.0;
    // b.grad_ = -4.0;
    std::cout << a->getValue() << std::endl;
    std::cout << b->getValue() << std::endl;
   
    c->label_ = "c";
    // c.grad_ = -2.0;
    std::cout << c->getValue() << std::endl;
 
    auto e = a * b;
    e->label_ = "e";
    // e.grad_ = -2.0;
    std::cout << e->getValue() << std::endl;
    
    auto d = e + c;
    d->label_ = "d";
    // d.grad_ = -2.0;
    std::cout << d->getValue() << std::endl;


    auto f = std::make_shared<Value>(-2.0, std::vector<std::shared_ptr<Value>>{}, "", "f");

    // f.grad_ = 4.0;
    std::cout << f->getValue() << std::endl;

    auto L = d * f;
    L->label_ = "L";
    // L.grad_ = 1.0;
    std::cout << L->getValue() << std::endl;
    
    
    // inputs x1, x2
    auto x1 = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "x1");
    auto x2 = std::make_shared<Value>(0.0, std::vector<std::shared_ptr<Value>>{}, "", "x2");

    auto w1 = std::make_shared<Value>(-3.0, std::vector<std::shared_ptr<Value>>{}, "", "w1");
    auto w2 = std::make_shared<Value>(1.0, std::vector<std::shared_ptr<Value>>{}, "", "w2");

    // x2.grad_ = 0.5;
    // w2.grad_ = 0.0;

    // x1.grad_ = -1.5;
    // w1.grad_ = 1.0;


    // bias b
    auto b1 = std::make_shared<Value>(6.8813735870195431, std::vector<std::shared_ptr<Value>>{}, "", "b1");

    // b1.grad_ = 0.5;
    

    // x1 * w1 + 2*w2 +b
    auto x1w1 = x1 * w1;
    x1w1->label_ = "x1w1";
    // x1w1.grad_ = 0.5;
   
    auto x2w2 = x2 * w2;
    x2w2->label_ = "x2w2";
    // x2w2.grad_ = 0.5;

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label_ = "x1w1x2w2";
    // x1w1x2w2.grad_ = 0.5;
   

    auto n = x1w1x2w2 + b1;
    n->label_ = "n";
    // n.grad_ = 0.5;


    auto o = n->tanh();
    std::cout << "external n address: " << &n << std::endl;
    o->label_ = "o";
    o->grad_ = 1.0;
    std::cout << "o data: " << o->data_ << std::endl;
    o->_backward();
    n->_backward();
    b1->_backward();
    x1w1x2w2->_backward();
    x1w1->_backward();
    x2w2->_backward();

    std::cout << "Backward pass:" << std::endl;
    std::cout << x1->getValue() << std::endl;
    std::cout << x2->getValue() << std::endl;
    std::cout << w1->getValue() << std::endl;
    std::cout << w2->getValue() << std::endl;
    std::cout << b1->getValue() << std::endl;
    std::cout << x1w1->getValue() << std::endl;
    std::cout << x2w2->getValue() << std::endl;
    std::cout << x1w1x2w2->getValue() << std::endl;
    std::cout << n->getValue() << std::endl;
    std::cout << o->getValue() << std::endl;


    
    
    
    return 0;
}