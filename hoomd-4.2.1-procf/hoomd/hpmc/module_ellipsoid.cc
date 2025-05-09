// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// ########## Modified by Rheoinformatic //~ [RHEOINF] ##########

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "../Variant.h" //~ add vinf [RHEOINF]

#include "ComputeSDF.h"
#include "ShapeEllipsoid.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"

#include "ShapeMoves.h"
#include "UpdaterShape.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterClustersGPU.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Export the base HPMCMono integrators
void export_ellipsoid(pybind11::module& m)
    {
    //~ Update the function calls to pass both required arguments [RHEOINF]
    m.def("create_IntegratorHPMCMonoEllipsoid", [](std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Variant> vinf)
    {
        return std::make_shared<IntegratorHPMCMono<ShapeEllipsoid>>(sysdef, vinf);
    });
    //export_IntegratorHPMCMono<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoid");
    //~
    export_ComputeFreeVolume<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoid");
    export_ComputeSDF<ShapeEllipsoid>(m, "ComputeSDFEllipsoid");
    export_UpdaterMuVT<ShapeEllipsoid>(m, "UpdaterMuVTEllipsoid");
    export_UpdaterClusters<ShapeEllipsoid>(m, "UpdaterClustersEllipsoid");

    export_MassProperties<ShapeEllipsoid>(m, "MassPropertiesEllipsoid");

    export_UpdaterShape<ShapeEllipsoid>(m, "UpdaterShapeEllipsoid");
    export_ShapeMoveBase<ShapeEllipsoid>(m, "ShapeMoveBaseShapeEllipsoid");
    export_PythonShapeMove<ShapeEllipsoid>(m, "ShapeSpaceEllipsoid");
    export_ElasticShapeMove<ShapeEllipsoid>(m, "ElasticEllipsoid");

    export_ExternalFieldInterface<ShapeEllipsoid>(m, "ExternalFieldEllipsoid");
    export_HarmonicField<ShapeEllipsoid>(m, "ExternalFieldHarmonicEllipsoid");
    export_ExternalFieldWall<ShapeEllipsoid>(m, "WallEllipsoid");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoidGPU");
    export_ComputeFreeVolumeGPU<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoidGPU");
    export_UpdaterClustersGPU<ShapeEllipsoid>(m, "UpdaterClustersEllipsoidGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
