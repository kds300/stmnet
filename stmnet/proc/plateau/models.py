""" Process model for the NxPlateau neuron.
"""

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from stmnet.proc.plateau.process import NxPlateau


@implements(proc=NxPlateau, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyNxPlateauModelFixed(PyLoihiProcessModel):
    """ Implementation of Plateau neuron process in fixed point precision.

    Precisions of state variables

    - dv : unsigned 12-bit integer (0 to 4095)
    - du : unsigned 12-bit integer (0 to 4095)
    - vth : unsigned 17-bit integer (0 to 131071)
    - uth : unsigned 17-bit integer (0 to 131071)
    """

    a_soma_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    a_dend_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    v_soma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v_dend: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    dv_soma: int = LavaPyType(int, np.uint16, precision=12)
    dv_dend: int = LavaPyType(int, np.uint16, precision=12)
    vth_soma: int = LavaPyType(int, np.int32, precision=17)
    vth_dend: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyNxPlateauModelFixed, self).__init__(proc_params)
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        self.decay_shift = 12
        self.decay_unity = 2 ** self.decay_shift
        self.vth_shift = 6
        self.act_shift = 6
        self.isthrscaled = False

    def scale_threshold(self):
        self.effective_vth_dend = np.left_shift(self.vth_dend, self.vth_shift)
        self.effective_vth_soma = np.left_shift(self.vth_soma, self.vth_shift)
        self.isthrscaled = True

    def subthr_dynamics(
        self,
        activation_dend_in: np.ndarray,
        activation_soma_in: np.ndarray
    ):
        """Run the sub-threshold dynamics for both the dendrite and soma of the
        neuron. Both use 'leaky integration'.
        """
        for v, dv, a_in in [
            (self.v_dend, self.dv_dend, activation_dend_in),
            (self.v_soma, self.dv_soma, activation_soma_in),
        ]:
            decayed_volt = np.int64(v) * (self.decay_unity - dv)
            decayed_volt = np.sign(decayed_volt) * np.right_shift(
                np.abs(decayed_volt), 12
            )
            decayed_volt = np.int32(decayed_volt)
            updated_volt = decayed_volt + np.left_shift(a_in, self.act_shift)

            neg_voltage_limit = -np.int32(self.max_uv_val) + 1
            pos_voltage_limit = np.int32(self.max_uv_val) - 1

            v[:] = np.clip(
                updated_volt, neg_voltage_limit, pos_voltage_limit
            )

    def soma_spike_and_reset(self):
        """Check the spiking conditions for the plateau soma. Checks if:
            v_soma > v_th_soma
            up_state > 0

        For any neurons n that satisfy both conditions, sets:
            s_out_buff[n] = True
            v_soma = 0
        """
        s_out_buff = np.logical_and(
            self.v_soma > self.effective_vth_soma,
            self.v_dend > self.effective_vth_dend
        )
        self.v_soma[s_out_buff] = 0

        return s_out_buff

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated bgy a PyLoihiProcessModel using the
        LoihiProtocol.
        """

        # Receive synaptic input
        a_dend_in_data = self.a_dend_in.recv()
        a_soma_in_data = self.a_soma_in.recv()

        # Check threshold scaling
        if not self.isthrscaled:
            self.scale_threshold()

        self.subthr_dynamics(a_dend_in_data, a_soma_in_data)

        self.s_out_buff = self.soma_spike_and_reset()

        self.s_out.send(self.s_out_buff)
