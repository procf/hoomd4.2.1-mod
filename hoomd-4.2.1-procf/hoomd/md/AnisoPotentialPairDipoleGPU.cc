// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairDipole.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairDipoleGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairDipole>(m, "AnisoPotentialPairDipoleGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
