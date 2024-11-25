"""Custom process modelling dendritic plateaus.
"""

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class NxPlateau(AbstractProcess):
    """Nx-style dendritic plateau neuron."""
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 dv_soma: float,
                 dv_dend: float,
                 vth_soma: float,
                 vth_dend: float,
                 name: ty.Optional[str],
                ):
        super().__init__(shape=shape,
                         dv_soma=dv_soma,
                         dv_dend=dv_dend,
                         name=name,
                         vth_soma=vth_soma,
                         vth_dend=vth_dend,
                        )
        self.a_soma_in = InPort(shape=shape)
        self.a_dend_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v_soma = Var(shape=shape, init=0)
        self.v_dend = Var(shape=shape, init=0)
        self.dv_soma = Var(shape=(1,), init=dv_soma)
        self.dv_dend = Var(shape=(1,), init=dv_dend)
        self.vth_soma = Var(shape=(1,), init=vth_soma)
        self.vth_dend = Var(shape=(1,), init=vth_dend)

