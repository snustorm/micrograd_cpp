#include "ops.h"

std::shared_ptr<Value> operator+(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    auto out = std::make_shared<Value>(
        lhs->data_ + rhs->data_, std::vector{lhs, rhs}, "+"
    );

    out->_backward = [lhs, rhs, out]() {
        lhs->grad_ += 1.0 * out->grad_;
        rhs->grad_ += 1.0 * out->grad_;
    };

    return out;
}


std::shared_ptr<Value> operator*(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    auto out = std::make_shared<Value>(
        lhs->data_ * rhs->data_, std::vector{lhs, rhs}, "*"
    );

    out->_backward = [lhs, rhs, out]() {
        lhs->grad_ += rhs->data_ * out->grad_;
        rhs->grad_ += lhs->data_ * out->grad_;
    };

    return out;
}


std::shared_ptr<Value> operator-(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    auto out = std::make_shared<Value>(
        lhs->data_ - rhs->data_, std::vector{lhs, rhs}, "-"
    );

    out->_backward = [lhs, rhs, out]() {
        lhs->grad_ += 1.0 * out->grad_;
        rhs->grad_ -= 1.0 * out->grad_;
    };

    return out;
}