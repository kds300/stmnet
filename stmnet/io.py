"""STMNet Input/Output Processes
"""

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.io.source import RingBuffer as Source
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

import stmnet.proc as proc


class SpikeInputLayer(AbstractProcess):
    """Creates a spike generating layer to serve as input to a network."""
    def __init__(self, data: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        data: array
            binary array representing spikes.
            Expects shape (n_ports, n_timesteps)
        """
        shape = (data.shape[0],)
        super().__init__(data=data, shape=shape, **kwargs)

        self.s_out = OutPort(shape=shape)

        self.data = Var(shape=data.shape, init=data)


@implements(proc=SpikeInputLayer, protocol=LoihiProtocol)
class SubSpikeInputLayerModel(AbstractSubProcessModel):
    """Implements the SpikeInputLayer."""
    def __init__(self, proc):
        """Builds spike source using a RingBuffer."""
        data = proc.proc_params.get("data")

        self.source = Source(data=data)

        self.source.out_ports.s_out.connect(proc.out_ports.s_out)
        
        proc.vars.data.alias(self.source.vars.data)


def create_binary_io_processes(
    binary_input_spikes, template_input_spikes, layer_keys, runtime
):
    """Creates io processes for the binary module, assuming template spikes
    come from a single layer.

    Can be used to run the Binary module by itself, simulating template spikes.
    """
    # init data arrays
    binary_input_data = np.zeros(
        shape=(len(layer_keys['binary']), runtime)
    )
    template_input_data = np.zeros(
        shape=(len(layer_keys['template']), runtime)
    )

    # add spikes
    for (i, t) in binary_input_spikes:
        binary_input_data[i, t] = 1
    for (i, t) in template_input_spikes:
        template_input_data[i, t] = 1

    binary_sg = proc.SpikeIn(data=binary_input_data)
    template_sg = proc.SpikeIn(data=template_input_data)

    # decision output
    decision_sr = proc.SpikeOut(
        shape=(len(layer_keys['syllable']),),
        buffer=runtime
    )

    return binary_sg, template_sg, decision_sr

def create_spike_output(spikes_out_port, runtime):
    """Create a spike output process, connecting it to specified out port."""
    s_out = proc.SpikeOut(shape=spikes_out_port.shape, buffer=runtime)
    spikes_out_port.connect(s_out.a_in)

    return s_out
