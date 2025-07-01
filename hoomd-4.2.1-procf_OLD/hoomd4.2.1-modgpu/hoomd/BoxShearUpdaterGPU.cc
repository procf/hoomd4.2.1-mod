//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan


// ########## Created by Rheoinformatic //~ [RHEOINF] ##########

/*! \file BoxShearUpdaterGPU.cc
    \brief Defines the BoxShearUpdaterGPU class using the GPU
*/

#include "BoxShearUpdaterGPU.h"
#include "BoxShearUpdaterGPU.cuh"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxShearUpdaterGPU::BoxShearUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Trigger> trigger,
                                       std::shared_ptr<Variant> vinf,
                                       Scalar deltaT,
                                       bool flip,
                                       std::shared_ptr<ParticleGroup> group)
    : BoxShearUpdater(sysdef, trigger, vinf, deltaT, flip, group)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot initialize BoxShearUpdaterGPU on a CPU device.");
        }
    m_tuner_wrap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "box_shear_wrap"));
    }

BoxShearUpdaterGPU::~BoxShearUpdaterGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxShearUpdaterGPU" << endl;
    }

void BoxShearUpdaterGPU::update(uint64_t timestep)
    {
    // Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Box shear update GPU" << endl;

    BoxDim cur_box = m_pdata->getGlobalBox();
    Scalar L_Y = cur_box.getL().y;

    Scalar cur_erate = (*m_vinf)(timestep)/L_Y;
    Scalar3 new_L = cur_box.getL();
    Scalar xy = cur_box.getTiltFactorXY() + cur_erate * m_deltaT;
    
    if(m_flip)
        {
        if(xy>Scalar(0.50)) xy -= Scalar(1.0);
        else if(xy<Scalar(-0.50)) xy += Scalar(1.0);
        }
    Scalar xz = cur_box.getTiltFactorXZ();
    Scalar yz = cur_box.getTiltFactorYZ();
    BoxDim new_box = BoxDim(new_L);
    new_box.setTiltFactors(xy, xz, yz);


    if (new_box != cur_box)
        {
        m_pdata->setGlobalBox(new_box);

        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::readwrite);

        // const BoxDim& local_box = m_pdata->getBox();
        m_tuner_wrap->begin();
        kernel::gpu_box_shear_wrap(m_pdata->getN(),
                                   m_pdata->getBox(),
                                   L_Y,
                                   d_pos.data,
                                   d_image.data,
                                   d_vel.data,
                                   cur_erate,
                                   m_tuner_wrap->getParam()[0]);
        m_tuner_wrap->end();
        }

    } /* end of update*/

namespace detail
    {
void export_BoxShearUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<BoxShearUpdaterGPU, BoxShearUpdater, std::shared_ptr<BoxShearUpdaterGPU>>(
        m,
        "BoxShearUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<Variant>,Scalar, bool, std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail

    } // end namespace hoomd
