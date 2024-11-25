"""STMNet Input/Output Processes
"""

import numpy as np

import stmnet.proc as proc


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
