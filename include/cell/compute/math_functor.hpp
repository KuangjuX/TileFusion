// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell::compute {

template <typename Element>
struct Add {
    DEVICE Element operator()(Element a, Element b) const { return a + b; }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs + rhs;
    }
};

template <typename Element>
struct Sub {
    DEVICE Element operator()(Element a, Element b) const { return a - b; }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs - rhs;
    }
};

template <typename Element>
struct Mul {
    DEVICE Element operator()(Element a, Element b) const { return a * b; }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs * rhs;
    }
};

template <typename Element>
struct Div {
    DEVICE Element operator()(Element a, Element b) const { return a / b; }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs / rhs;
    }
};

template <typename Element>
struct Max {
    DEVICE Element operator()(Element a, Element b) const {
        return a > b ? a : b;
    }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs > rhs ? lhs : rhs;
    }
};

template <typename Element>
struct Min {
    DEVICE Element operator()(Element a, Element b) const {
        return a < b ? a : b;
    }

    DEVICE void operator()(const Element& lhs, const Element& rhs,
                           Element& dst) {
        dst = lhs < rhs ? lhs : rhs;
    }
};

template <typename Element>
struct Exp {
    DEVICE Element operator()(Element a) const { return exp(a); }

    DEVICE void operator()(const Element& src, Element& dst) { dst = exp(src); }
};

#if defined(__CUDA_ARCH__)
template <>
struct Exp<float> {
    DEVICE float operator()(float a) const { return __expf(a); }

    DEVICE void operator()(const float& src, float& dst) { dst = __expf(src); }
};

template <>
struct Exp<__half> {
    DEVICE __half operator()(__half a) const { return hexp(a); }

    DEVICE void operator()(const __half& src, __half& dst) { dst = hexp(src); }
};
#endif

template <typename Element>
struct Log {
    DEVICE Element operator()(Element a) const { return log(a); }

    DEVICE void operator()(const Element& src, Element& dst) { dst = log(src); }
};

#if defined(__CUDA_ARCH__)
template <>
struct Log<float> {
    DEVICE float operator()(float a) const { return __logf(a); }

    DEVICE void operator()(const float& src, float& dst) { dst = __logf(src); }
};

template <>
struct Log<__half> {
    DEVICE __half operator()(__half a) const { return hlog(a); }

    DEVICE void operator()(const __half& src, __half& dst) { dst = hlog(src); }
};
#endif

template <typename Element>
struct Relu {
    DEVICE Element operator()(Element a) const { return a > 0 ? a : 0; }

    DEVICE void operator()(const Element& src, Element& dst) {
        dst = src > 0 ? src : 0;
    }
};

#if defined(__CUDA_ARCH__)
template <>
struct Relu<float> {
    DEVICE float operator()(float a) const { return max(a, 0.f); }

    DEVICE void operator()(const float& src, float& dst) {
        dst = max(src, 0.f);
    }
};

template <>
struct Relu<__half> {
    DEVICE __half operator()(__half a) const { return __hmax(a, 0); }

    DEVICE void operator()(const __half& src, __half& dst) {
        dst = __hmax(src, 0);
    }
};
#endif

template <typename SrcType, typename DstType>
struct Convert {
    DEVICE DstType operator()(SrcType a) const {
        return static_cast<DstType>(a);
    }

    DEVICE void operator()(const SrcType& src, DstType& dst) {
        dst = static_cast<DstType>(src);
    }
};

#if defined(__CUDA_ARCH__)

template <>
struct Convert<float, __half> {
    DEVICE __half operator()(float a) const { return __float2half(a); }

    DEVICE void operator()(const float& src, __half& dst) {
        dst = __float2half(src);
    }
};

template <>
struct Convert<__half, float> {
    DEVICE float operator()(__half a) const { return __half2float(a); }

    DEVICE void operator()(const __half& src, float& dst) {
        dst = __half2float(src);
    }
};
#endif

}  // namespace tilefusion::cell::compute
