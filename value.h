#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <functional>



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
    
    
    std::string getValue() const { 
        std::ostringstream oss;
        oss << "Value(" << label_
            << " | data = " << std::fixed << std::setprecision(4) << data_
            << ", grad = " << std::fixed << std::setprecision(4) << grad_
            << ")";
        return oss.str();
    }

    std::shared_ptr<Value> operator+( Value& other) const {
        
        auto self = std::const_pointer_cast<Value>(shared_from_this()); 
        
        auto others = std::make_shared<Value>(other);

        auto out = std::make_shared<Value>(data_ + other.data_, std::vector{self, others}, "+");

        
        out->_backward = [self, others, grad = out->grad_]() {
            self->grad_ += 1.0 * grad;
            others->grad_ += 1.0 * grad;
        };

        return out;
    }

    std::shared_ptr<Value> operator*(const Value& other) const {
        auto self = std::make_shared<Value>(*this);
        auto others = std::make_shared<Value>(other);

        //Value out(data_ * other.data_, {self, others}, "*");
        auto out = std::make_shared<Value>(data_ * other.data_, std::vector{self, others}, "*");

        out->_backward = [self, others, grad = out->grad_]() {
            self->grad_ += others->data_ * grad;
            others->grad_ += self->data_ * grad;
        };

        return out;
    }
    
    Value operator-(const Value& other) const {
        auto self = std::make_shared<Value>(*this);
        auto others = std::make_shared<Value>(other);

        Value out(data_ - other.data_, {self, others}, "-");

        out._backward = [self, others, grad = out.grad_]() {
            self->grad_ += 1.0 * grad;
            others->grad_ -= 1.0 * grad;
        };

        return out;
    }
    
    std::shared_ptr<Value> tanh() const {
        
        auto self = std::const_pointer_cast<Value>(shared_from_this()); 
       
        auto out = std::make_shared<Value>(std::tanh(data_), std::vector{self}, "tanh");

        out->_backward = [self, out]() {
            self->grad_ += (1 - std::pow(std::tanh(self->data_), 2)) * out->grad_;
        };

        return out;
    }

    void printChildren() const {
        std::cout << "{ Childrens: ";
        for (const auto& child : prev_) {
            std::cout << child->getValue() << " ";
        }
        std::cout << "}" << std::endl;
    }

    double data_;
    double grad_;   
    std::vector<std::shared_ptr<Value>> prev_;
    std::string op_;
    std::string label_;
    std::function<void()> _backward = [] () {};
    
};
