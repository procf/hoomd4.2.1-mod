########## Created by the Rheoinformatic research group ##########
# HOOMD-blue:
#   Copyright (c) 2009-2022 The Regents of the University of Michigan.
#   Part of HOOMD-blue, released under the BSD 3-Clause License.
#
# This file:
#   Written by Dr. Deepak Mangal 
#   Documentation by Rob Campbell (2024)

########## Created by Rheoinformatic ##~ [RHEOINF] ##########

"""Implement BoxResize."""

from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.trigger import Periodic


class BoxShear(Updater):

    def __init__(self, trigger, vinf, deltaT, flip, filter=All()):
        params = ParameterDict(vinf=Variant,
                               deltaT = float,
                               flip=bool,
                               filter=ParticleFilter)
        params['vinf'] = vinf
        params['trigger'] = trigger
        params['deltaT'] = deltaT
        params['flip'] = flip
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _hoomd.BoxShearUpdater(
            self._simulation.state._cpp_sys_def, self.trigger, self.vinf, self.deltaT, self.flip, group)
        super()._attach_hook()
