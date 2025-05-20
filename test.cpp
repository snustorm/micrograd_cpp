#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <functional>
#include <set>

#include "ops.h"
#include "value.h"


// void build_topo(const std::shared_ptr<Value> & v,
//     std::set<std::shared_ptr<Value>> & visited,
//     std::vector<std::shared_ptr<Value>> & topo
// ){
//     if(visited.find(v) == visited.end())
//     {
//         visited.insert(v);
//         for (auto & child: v->prev_){
//             build_topo(child, visited, topo);
//         }
//         topo.push_back(v);
//     }
// }


int main() {
    std::cout << "Micrograd!" << std::endl;

    auto x = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "a");

    auto a = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "a");
    auto b = std::make_shared<Value>(-3.0, std::vector<std::shared_ptr<Value>>{}, "", "b");
    auto c = std::make_shared<Value>(10.0, std::vector<std::shared_ptr<Value>>{}, "", "c");

    std::cout << a->getValue() << std::endl;
    std::cout << b->getValue() << std::endl;
   
    c->label_ = "c";
    // c.grad_ = -2.0;
    std::cout << c->getValue() << std::endl;
 
    auto e = a * b;
    e->label_ = "e";
    std::cout << e->getValue() << std::endl;
    
    auto d = e + c;
    d->label_ = "d";
    std::cout << d->getValue() << std::endl;


    auto f = std::make_shared<Value>(-2.0, std::vector<std::shared_ptr<Value>>{}, "", "f");

    std::cout << f->getValue() << std::endl;

    auto L = d * f;
    L->label_ = "L";
    std::cout << L->getValue() << std::endl;
    
    
    // inputs x1, x2
    auto x1 = std::make_shared<Value>(2.0, std::vector<std::shared_ptr<Value>>{}, "", "x1");
    auto x2 = std::make_shared<Value>(0.0, std::vector<std::shared_ptr<Value>>{}, "", "x2");

    auto w1 = std::make_shared<Value>(-3.0, std::vector<std::shared_ptr<Value>>{}, "", "w1");
    auto w2 = std::make_shared<Value>(1.0, std::vector<std::shared_ptr<Value>>{}, "", "w2");

    // bias b
    auto b1 = std::make_shared<Value>(6.8813735870195431, std::vector<std::shared_ptr<Value>>{}, "", "b1");

    auto x1w1 = x1 * w1;
    x1w1->label_ = "x1w1";
   
    auto x2w2 = x2 * w2;
    x2w2->label_ = "x2w2";

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label_ = "x1w1x2w2";
   

    auto n = x1w1x2w2 + b1;
    n->label_ = "n";


    auto o = n->tanh();
    o->label_ = "o";
    o->grad_ = 1.0;

    o->backward();

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


    // std::cout << "Topological order of the graph:\n";
    // for (const auto& v : topo) {
    //     std::cout << "Label: " << v->label_
    //               << ", Data: " << v->data_
    //               << ", Grad: " << v->grad_
    //               << ", Op: " << v->op_
    //               << "\n";
    // }

    
    return 0;
}

