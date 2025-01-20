"""STMNet Input/Output Processes


Classes
-------
SpikeInputLayer
    Lava process for providing spike input layer to a network.
SpikeOutputLayer
    Lava process for receiving spike output from a network.
"""

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
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


class SpikeOutputLayer(AbstractProcess):
    """Creates a spike receiving layer to serve as output to a network."""
    def __init__(self, shape, buffer):
        """
        Parameters
        ----------
        shape: tuple
            Defines the shape of the spike receiver.
        buffer: int
            Number of timesteps for which to store spike data.
        """
        super().__init__(shape=shape, buffer=buffer)
        
        self.a_in =  InPort(shape=shape)

        self.buffer = Var(shape=(1,), init=buffer)
        buffer_shape = shape + (buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))


@implements(SpikeOutputLayer, protocol=LoihiProtocol)
class SubSpikeOutputLayerModel(AbstractSubProcessModel):
    """Implements the SpikeOutputLayer."""
    def __init__(self, proc):
        """Builds spike receiver using a RingBuffer."""
        shape = proc.proc_params.get("shape", (1,))
        buffer = proc.proc_params.get("buffer", 1)

        self.sink = Sink(shape=shape, buffer=buffer)

        proc.in_ports.a_in.connect(self.sink.in_ports.a_in)

        proc.vars.data.alias(self.sink.vars.data)


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
