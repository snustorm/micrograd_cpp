#include "value.h"

void build_topo(const std::shared_ptr<Value>& v,
                std::set<std::shared_ptr<Value>>& visited,
                std::vector<std::shared_ptr<Value>>& topo) {
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        for (auto& child : v->prev_) {
            build_topo(child, visited, topo);
        }
        topo.push_back(v);
    }
}

std::string Value::getValue() const { 
    std::ostringstream oss;
    oss << "Value(" << label_
        << " | data = " << std::fixed << std::setprecision(4) << data_
        << ", grad = " << std::fixed << std::setprecision(4) << grad_
        << ")";
    return oss.str();
}

std::shared_ptr<Value> Value::operator+( Value& other) const {
    
    auto self = std::const_pointer_cast<Value>(shared_from_this()); 
    
    auto others = std::make_shared<Value>(other);

    auto out = std::make_shared<Value>(data_ + other.data_, std::vector{self, others}, "+");

    
    out->_backward = [self, others, grad = out->grad_]() {
        self->grad_ += 1.0 * grad;
        others->grad_ += 1.0 * grad;
    };

    return out;
}

std::shared_ptr<Value> Value::operator*(const Value& other) const {
    auto self = std::make_shared<Value>(*this);
    auto others = std::make_shared<Value>(other);

    auto out = std::make_shared<Value>(data_ * other.data_, std::vector{self, others}, "*");

    out->_backward = [self, others, grad = out->grad_]() {
        self->grad_ += others->data_ * grad;
        others->grad_ += self->data_ * grad;
    };

    return out;
}

Value Value::operator-(const Value& other) const {
    auto self = std::make_shared<Value>(*this);
    auto others = std::make_shared<Value>(other);

    Value out(data_ - other.data_, {self, others}, "-");

    out._backward = [self, others, grad = out.grad_]() {
        self->grad_ += 1.0 * grad;
        others->grad_ -= 1.0 * grad;
    };

    return out;
}

std::shared_ptr<Value> Value::tanh() const {
        
    auto self = std::const_pointer_cast<Value>(shared_from_this()); 
       
    auto out = std::make_shared<Value>(std::tanh(data_), std::vector{self}, "tanh");

    out->_backward = [self, out]() {
        self->grad_ += (1 - std::pow(std::tanh(self->data_), 2)) * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> Value::exp() const {
        
    auto self = std::const_pointer_cast<Value>(shared_from_this()); 
       
    auto out = std::make_shared<Value>(std::exp(data_), std::vector{self}, "exp");

    out->_backward = [self, out]() {
        self->grad_ += out->data_ * out->grad_;
    };

    return out;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>, std::shared_ptr<Value>>
Value::pow(T number) const {
        
    auto self = std::const_pointer_cast<Value>(shared_from_this()); 

    std::string label = "**" + std::to_string(number);
       
    auto out = std::make_shared<Value>(std::pow(data_, number), std::vector{self}, label);

    out->_backward = [self, number, out]() {
        self->grad_ += number * (std::pow(self->data_, number - 1)) * out->grad_;
    };

    return out;
}

void Value::printChildren() const {
    std::cout << "{ Childrens: ";
    for (const auto& child : prev_) {
        std::cout << child->getValue() << " ";
    }
    std::cout << "}" << std::endl;
}


 void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo{};
    std::set<std::shared_ptr<Value>> visited{};
    auto self = std::const_pointer_cast<Value>(shared_from_this()); 

    build_topo(self, visited, topo);

    self->grad_ = 1.0;
    std::reverse(topo.begin(), topo.end());
    for (auto & node : topo)
    {
        node->_backward();
    }
}