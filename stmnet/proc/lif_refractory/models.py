"""Custom model implementing Refractory LIF neuron process.

Modified code from the lava LIF process models:
<https://github.com/lava-nc/lava>
"""

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFixed

from stmnet.proc.lif_refractory.process import LIFRefractory


@implements(proc=LIFRefractory, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLifRefractoryModelFixed(AbstractPyLifModelFixed):
    """Fixed point implementation of the LIFRefractory model.

    See Also
    --------
    Modified from the Lava source code for a LIF neuron.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)
    refractory_period_end: np.ndarray = LavaPyType(np.ndarray, int)
    refractory_period: int = LavaPyType(int, np.int16, precision=8)
    def __init__(self, proc_params):
        super(PyLifRefractoryModelFixed, self).__init__(proc_params)
        self.effective_vth = 0
        self.refractory_period = proc_params["refractory_period"]

    def scale_threshold(self):
        """***** From Lava source models *****
        Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """*****From lava source models*****
        Spike when voltage exceeds threshold.
        """
        return self.v > self.effective_vth

    def subthr_dynamics(self, activation_in: np.ndarray):
        """*****Modified from Lava source*****
        Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """

        # Update current
        # --------------
        decay_const_u = self.du + self.ds_offset
        # Below, u is promoted to int64 to avoid overflow of the product
        # between u and decay constant beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits)
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.sign(decayed_curr) * np.right_shift(
            np.abs(decayed_curr), self.decay_shift
        )
        decayed_curr = np.int32(decayed_curr)
        # Hardware left-shifts synaptic input for MSB alignment
        activation_in = np.left_shift(activation_in, self.act_shift)
        # Add synptic input to decayed current
        decayed_curr += activation_in
        # Check if value of current is within bounds of 24-bit. Overflows are
        # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
        # becomes k and -(2**23 + k) becomes -k
        wrapped_curr = np.where(
            decayed_curr > self.max_uv_val,
            decayed_curr - 2 * self.max_uv_val,
            decayed_curr,
        )
        wrapped_curr = np.where(
            wrapped_curr <= -self.max_uv_val,
            decayed_curr + 2 * self.max_uv_val,
            wrapped_curr,
        )
        self.u[:] = wrapped_curr

        non_refractory = self.refractory_period_end < self.time_step
        # Update voltage
        # --------------
        decay_const_v = self.dv + self.dm_offset

        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1
        # Decaying voltage similar to current. See the comment above to
        # understand the need for each of the operations below.
        decayed_volt = (
            np.int64(self.v[non_refractory])
            * (self.decay_unity - decay_const_v)
        )
        decayed_volt = np.sign(decayed_volt) * np.right_shift(
            np.abs(decayed_volt), self.decay_shift
        )
        decayed_volt = np.int32(decayed_volt)
        updated_volt = (
            decayed_volt
            + self.u[non_refractory]
            + self.effective_bias[non_refractory]
        )
        self.v[non_refractory] = np.clip(
            updated_volt, neg_voltage_limit, pos_voltage_limit
        )

    def process_spikes(self, spike_vector: np.ndarray):
        """*****From Lava source*****"""
        self.refractory_period_end[spike_vector] = (self.time_step
                                                    + self.refractory_period)
        super().reset_voltage(spike_vector)

    def run_spk(self):
        """*****Modified from Lava source *****
        The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.scale_bias()
        # # Compute effective bias and threshold only once, not every time-step
        # if not self.isbiasscaled:
        #     self.scale_bias()

        if not self.isthrscaled:
            self.scale_threshold()

        self.subthr_dynamics(activation_in=a_in_data)

        self.s_out_buff = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        # Set refractory period for spiked neurons
        self.process_spikes(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)
