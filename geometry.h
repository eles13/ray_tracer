#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>

namespace geometry {

    template<size_t DIM, typename T>
    struct vec {
        vec() { for (size_t i = DIM; i--; data_[i] = T()); }

        T &operator[](const size_t i) {
            assert(i < DIM);
            return data_[i];
        }

        const T &operator[](const size_t i) const {
            assert(i < DIM);
            return data_[i];
        }

    private:
        T data_[DIM];
    };

    typedef vec<3, float> Vec3f;
    typedef vec<4, float> Vec4f;

    template<typename T>
    struct vec<3, T> {
        vec() : x(T()), y(T()), z(T()) {}

        vec(T X, T Y, T Z) : x(X), y(Y), z(Z) {}

        T &operator[](const size_t i) {
            assert(i < 3);
            return i <= 0 ? x : (1 == i ? y : z);
        }

        const T &operator[](const size_t i) const {
            assert(i < 3);
            return i <= 0 ? x : (1 == i ? y : z);
        }

        const float norm() const { return std::sqrt(x * x + y * y + z * z); }

        vec<3, T> &normalize(T l = 1) {
            *this = (*this) * (l / norm());
            return *this;
        }

        T x, y, z;
    };

    template<typename T>
    struct vec<4, T> {
        vec() : x(T()), y(T()), z(T()), w(T()) {}

        vec(T X, T Y, T Z, T W) : x(X), y(Y), z(Z), w(W) {}

        T &operator[](const size_t i) {
            assert(i < 4);
            return i <= 0 ? x : (1 == i ? y : (2 == i ? z : w));
        }

        const T &operator[](const size_t i) const {
            assert(i < 4);
            return i <= 0 ? x : (1 == i ? y : (2 == i ? z : w));
        }

        T x, y, z, w;
    };

    template<size_t DIM, typename T>
    T operator*(const vec<DIM, T> &lhs, const vec<DIM, T> &rhs) {
        T ret = T();
        for (size_t i = DIM; i--; ret += lhs[i] * rhs[i]);
        return ret;
    }

    template<size_t DIM, typename T>
    vec<DIM, T> operator+(vec<DIM, T> lhs, const vec<DIM, T> &rhs) {
        for (size_t i = DIM; i--; lhs[i] += rhs[i]);
        return lhs;
    }

    template<size_t DIM, typename T>
    vec<DIM, T> operator-(vec<DIM, T> lhs, const vec<DIM, T> &rhs) {
        for (size_t i = DIM; i--; lhs[i] -= rhs[i]);
        return lhs;
    }

    template<size_t DIM, typename T, typename U>
    vec<DIM, T> operator*(const vec<DIM, T> &lhs, const U &rhs) {
        vec<DIM, T> ret;
        for (size_t i = DIM; i--; ret[i] = lhs[i] * rhs);
        return ret;
    }

    template<size_t DIM, typename T>
    vec<DIM, T> operator-(const vec<DIM, T> &lhs) {
        return lhs * T(-1);
    }

    template<typename T>
    vec<3, T> cross(vec<3, T> v1, vec<3, T> v2) {
        return vec<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }

    template<size_t DIM, typename T>
    std::ostream &operator<<(std::ostream &out, const vec<DIM, T> &v) {
        for (unsigned int i = 0; i < DIM; i++) out << v[i] << " ";
        return out;
    }

    template<size_t DIM, typename T>
    vec<DIM, T> operator/(vec<DIM, T> &&lhs, const float f) {
        return vec<DIM, T>(lhs.x / f, lhs.y / f, lhs.z / f);
    }

    template<size_t DIM, typename T>
    vec<DIM, T> reflect(const vec<DIM, T> &I, const vec<DIM, T> &N) {
        return (I - N * 2.f * (I * N) / (N.norm() * N.norm()));
    }

    template<size_t DIM, typename T>
    vec<DIM, T> refract(const vec<DIM, T> &I, const vec<DIM, T> &N, const float eta_t, const float eta_i = 1.f) {
        float cosi = -std::max(-1.f, std::min(1.f, I * N));
        if (cosi < 0) return refract(I, -N, eta_i, eta_t);
        float eta = eta_i / eta_t;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? Vec3f(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k));
    }

}

#endif //__GEOMETRY_H__
