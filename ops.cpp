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

std::shared_ptr<Value> operator+(
    const std::shared_ptr<Value>& lhs,
    double number
) {

    auto number_ptr = std::make_shared<Value>(number);

    auto out = std::make_shared<Value>(
        lhs->data_ + number_ptr->data_, std::vector{lhs, number_ptr}, "+"
    );

    out->_backward = [lhs, number_ptr, out]() {
        lhs->grad_ += 1.0 * out->grad_;
        number_ptr->grad_ += 1.0 * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> operator+(
    double lhs,
    const std::shared_ptr<Value>& rhs
) {
    return rhs + lhs;  // reuse the existing overload
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

std::shared_ptr<Value> operator*(
    const std::shared_ptr<Value>& lhs,
    double number
) {

    auto number_ptr = std::make_shared<Value>(number);
    auto out = std::make_shared<Value>(
        lhs->data_ * number_ptr->data_, std::vector{lhs, number_ptr}, "*"
    );

    out->_backward = [lhs, number_ptr, out]() {
        lhs->grad_ += number_ptr->data_ * out->grad_;
        number_ptr->grad_ += lhs->data_ * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> operator*(
    double lhs,
    const std::shared_ptr<Value>& rhs
) {
    return rhs * lhs;  
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
