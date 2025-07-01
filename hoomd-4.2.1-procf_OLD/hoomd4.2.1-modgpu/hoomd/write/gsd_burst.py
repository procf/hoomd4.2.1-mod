# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Write GSD last :math:`N` frames at user direction."""

from hoomd import _hoomd
from hoomd.filter import All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.write.gsd import GSD


class Burst(GSD):
    r"""Write the last :math:`N` stored frames in the GSD format.

    When triggered, `Burst` stores up to the last :math:`N` frames in a buffer.
    `Burst` writes the contents of the buffer to the file when `dump` is called.
    When the the next frame would result in :math:`N + 1` frames being stored,
    the oldest frame is removed and the new frame is added.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to store
            in the buffer.
        filename (str): File name to write.
        filter (hoomd.filter.filter_like): Select the particles to write.
            Defaults to `hoomd.filter.All`.
        mode (str): The file open mode. Defaults to ``'ab'``.
        dynamic (list[str]): Field names and/or field categores to save in
            all frames. Defaults to ``['property']``.
        logger (hoomd.logging.Logger): Provide log quantities to write. Defaults
            to `None`.
        max_burst_frames (int): The maximum number of frames to store before
            between writes. -1 represents no limit. Defaults to -1.
        write_at_start (bool): When ``True`` **and** the file does not exist or
            has 0 frames: write one frame with the current state of the system
            when `hoomd.Simulation.run` is called. Defaults to ``False``.

    Warning:
        `Burst` errors when attempting to create a file or writing to one with
        zero frames, unless ``write_at_start`` is ``True``.

    Note:
        When analyzing files created by `Burst`, generally the first frame is
        not associated with the call to `Burst.dump`.

    Note:
        For more tips and qualifications see the `hoomd.write.GSD`
        documentation.

    Attributes:
        filename (str): File name to write.
        trigger (hoomd.trigger.Trigger): Select the timesteps to store
            in the buffer.
        filter (hoomd.filter.filter_like): Select the particles to write.
        mode (str): The file open mode.
        dynamic (list[str]): Field names and/or field categores to save in
            all frames.
        max_burst_frames (int): The maximum number of frames to store before
            between writes. -1 represents no limit.
        write_diameter (bool): When `False`, do not write
            ``particles/diameter``. Set to `True` to write non-default particle
            diameters.
        write_at_start (bool): When ``True`` **and** the file does not exist or
            has 0 frames: write one frame with the current state of the system
            when `hoomd.Simulation.run` is called.
    """

    def __init__(self,
                 trigger,
                 filename,
                 filter=All(),
                 mode='ab',
                 dynamic=None,
                 logger=None,
                 max_burst_size=-1,
                 write_at_start=False):
        super().__init__(trigger=trigger,
                         filename=filename,
                         filter=filter,
                         mode=mode,
                         dynamic=dynamic,
                         logger=logger)
        self._param_dict.pop("truncate")
        self._param_dict.update(
            ParameterDict(max_burst_size=int, write_at_start=bool))
        self._param_dict.update({
            "max_burst_size": max_burst_size,
            "write_at_start": write_at_start
        })

    def _attach_hook(self):
        sim = self._simulation
        self._cpp_obj = _hoomd.GSDDequeWriter(sim.state._cpp_sys_def,
                                              self.trigger, self.filename,
                                              sim.state._get_group(self.filter),
                                              self.logger, self.max_burst_size,
                                              self.mode, self.write_at_start,
                                              sim.timestep)

    def dump(self):
        """Write all currently stored frames to the file and empties the buffer.

        This method alllows for custom writing of frames at user specified
        conditions.
        """
        if self._attached:
            self._cpp_obj.dump()

    def __len__(self):
        """Get the current length of the internal frame buffer."""
        if self._attached:
            return len(self._cpp_obj)
        return 0
