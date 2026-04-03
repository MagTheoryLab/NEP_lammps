#pragma once

using StructReal = float;

#ifdef MAIN_NEP_SPIN_DOUBLE
using SpinReal = double;
#define MAIN_NEP_SPIN_PRECISION_NAME "double"
#else
using SpinReal = float;
#define MAIN_NEP_SPIN_PRECISION_NAME "float"
#endif

#define MAIN_NEP_STRUCT_PRECISION_NAME "float"
