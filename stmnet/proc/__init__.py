"""
Spiking Network Process Library
===============================

Collection of the Lava processes, both standard and custom, used in building the STMNet.
For more information about Lava processes, refer to the
`lava documentation <https://lava-nc.org/>`_
"""

from lava.proc.lif.process import LIF, LogConfig
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.proc.io.sink import RingBuffer as SpikeOut
from lava.proc.sparse.process import Sparse, DelaySparse

from stmnet.proc.plateau.process import NxPlateau
from stmnet.proc.lif_refractory.process import LIFRefractory
