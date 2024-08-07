#pragma once

#include "device_functions.h"

__device__
inline
int atomicCAS(int* address, int compare, int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned int atomicCAS(
    unsigned int* address, unsigned int compare, unsigned int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned long long atomicCAS(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}

__device__
inline
int atomicAdd(int* address, int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAdd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicAdd_impl(float* address, float val)
{
    unsigned int* uaddr{reinterpret_cast<unsigned int*>(address)};
    unsigned int r{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};

    unsigned int old;
    do {
        old = __atomic_load_n(uaddr, __ATOMIC_RELAXED);

        if (r != old) { r = old; continue; }

        r = atomicCAS(uaddr, r, __float_as_uint(val + __uint_as_float(r)));

        if (r == old) break;
    } while (true);

    return __uint_as_float(r);
}
#if !__has_builtin(__builtin_amdgcn_is_shared)
    __device__
    inline
    bool __builtin_amdgcn_is_shared(
        const __attribute__((address_space(0))) void* ptr) noexcept
    {
        #if defined(__HIP_DEVICE_COMPILE__)
            const unsigned int gp = reinterpret_cast<unsigned long long>(ptr);

            return gp ==
                (__builtin_amdgcn_s_getreg((15 << 11) | (16 << 6) | 15) << 16);
        #else
            return false;
        #endif
    }
#endif
__device__
inline
float atomicAdd(float* address, float val)
{
    using GP = const __attribute__((address_space(0))) void*;
    using LP = __attribute__((address_space(3))) float*;

    #if __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__ || __HIP_ARCH_GFX908__
        if (__builtin_amdgcn_is_shared((GP) address)) {
            return __builtin_amdgcn_ds_faddf((LP) address, val, 0, 0, false);
        }
    #endif

    return atomicAdd_impl(address, val);
}
__device__
inline
double atomicAdd(double* address, double val)
{
    unsigned long long* uaddr{reinterpret_cast<unsigned long long*>(address)};
    unsigned long long r{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};

    unsigned long long old;
    do {
        old = __atomic_load_n(uaddr, __ATOMIC_RELAXED);

        if (r != old) { r = old; continue; }

        r = atomicCAS(
            uaddr, r, __double_as_longlong(val + __longlong_as_double(r)));

        if (r == old) break;
    } while (true);

    return __longlong_as_double(r);
}

__device__
inline
int atomicSub(int* address, int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicExch(int* address, int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicExch(float* address, float val)
{
    return __uint_as_float(__atomic_exchange_n(
        reinterpret_cast<unsigned int*>(address),
        __float_as_uint(val),
        __ATOMIC_RELAXED));
}

__device__
inline
int atomicMin(int* address, int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}

__device__
inline
int atomicMax(int* address, int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_inc(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.inc.i32.p0i32");

    return __builtin_amdgcn_atomic_inc(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
    __device__
    extern
    unsigned int __builtin_amdgcn_atomic_dec(
        unsigned int*,
        unsigned int,
        unsigned int,
        unsigned int,
        bool) __asm("llvm.amdgcn.atomic.dec.i32.p0i32");

    return __builtin_amdgcn_atomic_dec(
        address, val, __ATOMIC_RELAXED, 1 /* Device scope */, false);
}

__device__
inline
int atomicAnd(int* address, int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAnd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicOr(int* address, int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicOr(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicXor(int* address, int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicXor(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}

// TODO: add scoped atomics i.e. atomic{*}_system && atomic{*}_block.
