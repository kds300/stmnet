"""Custom process implementing refractory period for LIF neurons.
"""

import typing as ty

import numpy as np
from lava.magma.core.process.process import LogConfig
from lava.magma.core.process.variable import Var
from lava.proc.lif.process import LIF


class LIFRefractory(LIF):
    """Leaky-Integrate-and-Fire (LIF) process with refractory period.

    Parameters
    ----------
    refractory_period : int, optional
        The interval of the refractory period. 1 timestep by default.

    See Also
    --------
    lava.proc.lif.process.LIF: 'Regular' leaky-integrate-and-fire neuron for
    documentation on rest of the behavior.

    `Lava docs<https://lava-nc.org>`_
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        refractory_period: ty.Optional[int] = 1,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
        )

        if refractory_period < 1:
            raise ValueError("Refractory period must be > 0.")

        self.proc_params["refractory_period"] = refractory_period
        self.refractory_period_end = Var(shape=shape, init=0)
        self.refractory_period = Var(shape=shape, init=refractory_period)
