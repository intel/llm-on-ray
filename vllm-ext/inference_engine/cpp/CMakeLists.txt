#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("inference_engine" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Option list
#

# general
option(IE_STATIC                 "inference_engine: static link libraries"                          OFF)
option(IE_NATIVE                 "inference_engine: enable -march=native flag"                      OFF)
option(IE_LTO                    "inference_engine: enable link time optimization"                  OFF)

# debug
option(IE_ALL_WARNINGS           "inference_engine: enable all compiler warnings"                   ON)
option(IE_ALL_WARNINGS_3RD_PARTY "inference_engine: enable all compiler warnings in 3rd party libs" OFF)
option(IE_GPROF                  "inference_engine: enable gprof"                                   OFF)

# sanitizers
option(IE_SANITIZE_THREAD        "inference_engine: enable thread sanitizer"                        OFF)
option(IE_SANITIZE_ADDRESS       "inference_engine: enable address sanitizer"                       OFF)
option(IE_SANITIZE_UNDEFINED     "inference_engine: enable undefined sanitizer"                     OFF)

# instruction set specific
option(IE_AVX                    "inference_engine: enable AVX"                                     ON)
option(IE_AVX2                   "inference_engine: enable AVX2"                                    ON)
option(IE_F16C                   "inference_engine: enable F16C"                                    ON)
option(IE_AVX512                 "inference_engine: enable AVX512"                                  OFF)
option(IE_AVX512_VBMI            "inference_engine: enable AVX512-VBMI"                             OFF)
option(IE_AVX512_VNNI            "inference_engine: enable AVX512-VNNI"                             OFF)
option(IE_FMA                    "inference_engine: enable FMA"                                     ON)
option(IE_AMX                    "inference_engine: enable AMX"                                     OFF)
option(IE_USE_OPENMP             "inference_engine: use OpenMP thread pool."                        ON)

option(IE_BUILD_TESTS            "inference_engine: build tests"                       ${IE_STANDALONE})
option(IE_BUILD_EXAMPLES         "inference_engine: build examples"                    ${IE_STANDALONE})
option(IE_USE_CLANG_TIDY         "inference_engine: clang-tidy check"                               OFF)


if(IE_BUILD_TESTS)
  add_compile_definitions(IE_BUILD_TESTS)
endif()
option(IE_PROFILING              "inference_engine: use Profiling"                                  OFF)
if (IE_PROFILING)
    add_compile_definitions(IE_PERF)
endif()
option(IE_BEAM_SEARCH_VERBOSE    "inference_engine: print beam search processing log"               OFF)
option(IE_GELU_VEC               "inference_engine: enable vec in gelu"                             ON)
if (IE_GELU_VEC)
    add_compile_definitions(IE_GELU_USE_VEC)
endif()
option(IE_SIMD_VEC_DOT_F16       "inference_engine: enable vec_dot_fp16 SIMD optimization"          ON)
option(BUILD_SHARED_LIBS         "If build as shared libs"                                       ON)

if (IE_SIMD_VEC_DOT_F16)
    add_compile_definitions(IE_SIMD_VEC_DOT_F16)
endif()

if(IE_BUILD_TESTS)
    enable_testing()
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)  # default to false so that pybind11 will not try to use IPO
if (IE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (NOT MSVC)
    if (IE_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (IE_GPROF)
        add_compile_options(-pg)
    endif()
    if (IE_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(IE_USE_OPENMP)
  include(FindOpenMP)
  # compile BesTLA's OMPTheading class, then it can be used in ne_layers
  set(BTLA_ENABLE_OPENMP ON CACHE BOOL "BesTLA enable compiling OpenMP threading")
  add_compile_definitions(IE_USE_OPENMP)
endif()

add_subdirectory(bestla)

include(cmake/ISA.cmake)
include(cmake/Common.cmake)
include(cmake/ClangTidy.cmake)

set(COMMON_HEADER_DIRS ./)
include_directories(${COMMON_HEADER_DIRS})


add_subdirectory(vectors)

add_subdirectory(core)

add_subdirectory(models)
