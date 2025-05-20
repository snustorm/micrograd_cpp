#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <functional>
#include <set>
#include <type_traits>

class Value : public std::enable_shared_from_this<Value>
{
public:
    Value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "", std::string label = "") 
        : data_(data), 
        grad_(0.0),
        prev_(children), 
        op_(op) ,
        label_(label)
        {}
    
    
    std::string getValue() const;

    std::shared_ptr<Value> operator+( Value& other) const;

    std::shared_ptr<Value> operator*(const Value& other) const;
    
    Value operator-(const Value& other) const;
    
    std::shared_ptr<Value> tanh() const;
    
    std::shared_ptr<Value> exp() const;

    template<typename T>
    std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>, std::shared_ptr<Value>>
    pow(T number) const;

    void printChildren() const;

    void backward();


    double data_;
    double grad_;   
    std::vector<std::shared_ptr<Value>> prev_;
    std::string op_;
    std::string label_;
    std::function<void()> _backward = [] () {};
    
};


void build_topo(const std::shared_ptr<Value> & v,
    std::set<std::shared_ptr<Value>> & visited,
    std::vector<std::shared_ptr<Value>> & topo);
