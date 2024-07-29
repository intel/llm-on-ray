#pragma once

#include <cstdint>

const int _TORCH_INT8_TYPE      = 3;
const int _TORCH_INT16_TYPE     = 4;
const int _TORCH_INT32_TYPE     = 5;
const int _TORCH_INT64_TYPE     = 6;
const int _TORCH_UINT8_TYPE     = 7;

template<int i> class DataArray {};

template<> class DataArray<_TORCH_INT8_TYPE> { typedef int8_t * type; };
template<> class DataArray<_TORCH_INT16_TYPE> { typedef int16_t * type; };
template<> class DataArray<_TORCH_INT32_TYPE> { typedef int32_t * type; };
template<> class DataArray<_TORCH_INT64_TYPE> { typedef int64_t * type; };
template<> class DataArray<_TORCH_UINT8_TYPE> { typedef uint8_t * type; };