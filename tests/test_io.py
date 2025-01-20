"""Unit Tests for IO Module."""

import unittest

import numpy as np
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink

from stmnet.io import SpikeInputLayer, SpikeOutputLayer


class TestSpikeInputLayer(unittest.TestCase):
    """Unit Tests for the SpikeInputLayer hierarchical process."""
    def test_cpu_execution(self):
        """Test that SpikeInputLayer behaves correctly for CPU execution.
        """
        N = 5
        data = np.eye(N)

        source = SpikeInputLayer(data=data)
        sink = Sink(shape=(N,), buffer=N)
        source.s_out.connect(sink.a_in)

        source.run(
            condition=RunSteps(num_steps=N),
            run_cfg=Loihi2SimCfg()
        )

        out_data = sink.data.get()

        source.stop()

        self.assertTrue(np.array_equal(data, out_data))


class TestSpikeOutputLayer(unittest.TestCase):
    """Unit Tests for the SpikeOutputLayer hierarchical process."""
    def test_cpu_execution(self):
        """Test that SpikeOutputLayer behaves correctly for CPU execution.
        """
        N = 5
        data = np.eye(N)

        source = Source(data=data)
        sink = SpikeOutputLayer(shape=(N,), buffer=N)
        source.s_out.connect(sink.a_in)

        sink.run(
            condition=RunSteps(num_steps=N),
            run_cfg=Loihi2SimCfg()
        )

        out_data = sink.data.get()

        sink.stop()

        self.assertTrue(np.array_equal(data, out_data))


if __name__ == "__main__":
    unittest.main()
