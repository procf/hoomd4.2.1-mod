#ifndef __ANGLEMAP_H__
#define __ANGLEMAP_H__

// AngleMap.h
#pragma once

#include <map>
// paniz_write
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include "hoomd/GlobalArray.h"
#include "hoomd/HOOMDMath.h"
#include <iostream>
#include <memory>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

// Declare the angle_map as an extern variable
namespace hoomd
{
    extern std::map<unsigned int, Scalar> angle_map_temp2;

}
#endif //__ANGLEMAP_H__
