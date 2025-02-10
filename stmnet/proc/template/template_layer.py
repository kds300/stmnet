"""Template Layer

Non-Lava class for building a Spiking Template Layer. Designed to mimic the
interface of a Lava process.

"""

import numpy as np

from stmnet.proc import LIFRefractory, Sparse, DelaySparse


class TemplateLayer:
    """Hierarchical-process-like class for a spike template matching layer.

    Templates are encoded as delayed sparse connections to a layer of LIF
    neurons, with each neuron representing a template.

    Attributes
    ----------
    shape: tuple
        The shape of the network should be (n_inputs, n_templates)
    s_in: list[InPort]
        List of the input ports for the sparse synapse layers of the network
    s_out: list[OutPort]
        List of the output ports for the LIF neurons in the network
    vth: array-like
        Voltage thresholds for the template neurons
    weights: array-like
        weight matrix for the template encoding
    delays: array-like
        delay matrix for the template encoding
    lif: list[LifProcess]
        Neuron processes for the layer
    sparse: list[DelaySparseProcess]
        Synapse processes for the layer
    out_conn: list[Sparse | DelaySparse]
        Synapses connecting from the LIF neurons to receiving processes
    """
    def __init__(
        self,
        delays,
        f=0.99,
        weights=1,
        vth=None,
        du=4095,
        dv=4095,
        refractory_period=1
    ):
        """
        Parameters
        ----------
        delays: array-like
            delay matrix encoding the templates
        f: float, default=1.0
            Fraction of template required to trigger a template match
        weights: int or array-like, default=1
            Weights to apply to synapse connections.
            If an integer, all connections will be given the same weight.
        vth: array-like, optional
            Voltage thresholds to apply to template neurons.
            If not supplied, will be calculated using shape, f, and weights
        du: int, default=4095
            Current decay parameter for template neurons. Default is to decay
            completely at each timestep.
        dv: int, default=4095
            Voltage decay parameter for template neurons. Default is to decay
            completely at each timestep.
        refractory_period: int, default=1
            Inacive period after spiking for the LIF neurons
        """
        # check that weights is valid
        if not isinstance(weights, int):
            raise NotImplementedError("Only integer weights implemented.")

        # determine vth values
        if vth is not None:
            raise NotImplementedError("Custom vth values not implemented.")

        n_dets_per_tpt = np.sum(
            np.where(delays, 1, 0),
            axis=1
        )
        vth = np.array([
            int(np.max(
                [
                    1,
                    int(
                        f
                        * weights
                        * n_dets
                    )
                ]
            ))
            for n_dets in n_dets_per_tpt
        ])

        # TODO: implement normalize_syllables

        self.lif = [
            LIFRefractory(
                shape=(1,),
                vth=vth_val,
                du=du,
                dv=dv,
                refractory_period=refractory_period
            )
            for vth_val in vth
        ]


        # set up the synapses
        weights = np.where(delays, weights, 0)

        self.sparse = [
            DelaySparse(
                weights=weights[i, :][np.newaxis, :],
                delays=delays[i, :][np.newaxis, :]
            )
            for i in range(delays.shape[0])
        ]


        # connect sparse -> lif
        # for _sparse, _lif in zip(self.sparse, self.lif):
        #     _sparse.a_out.connect(_lif.a_in)
        for i in range(len(self.lif)):
            self.sparse[i].a_out.connect(self.lif[i].a_in)


        # set up attributes
        self.s_in = [
            _sparse.s_in for _sparse in self.sparse
        ]
        self.s_out = [
            _lif.s_out for _lif in self.lif
        ]
        self.shape = (len(self.s_in), len(self.s_out))
        self.out_conn = []

        # set up aliases for execution
        self.run = self.lif[0].run
        self.stop = self.lif[0].stop

    def connect_sparse(
        self,
        port,
        weights,
        delays=None,
        connection_configs=None
    ) -> None:
        """Connects the LIF spike OutPorts to the supplied port using a list
        of sparse processes..

        Parameters
        ----------
        port: lava port
            Port to connect the LIF spike outputs to.
        weights: ndarray
            Weights for the connections. Expects an (port.shape[0], len(lif))
            array (treating the template LIF neurons as a single process)
        delays: ndarray, optional
            Delays for the LIF output connections. If None, Sparse process will
            be used instead of DelaySparse.
        connection_configs: optional
            Lava connection configs for the connections.
        """
        if weights.shape != (port.shape[0], len(self.lif)):
            raise ValueError(
                f"Weights shape {weights.shape} "
                f"does not equal {port.shape[0], len(self.lif)}"
            )

        if delays is None:
            for idx in range(self.shape[1]):
                _synapse = Sparse(
                    weights=weights[:, idx][:, np.newaxis]
                )
                self.lif[idx].s_out.connect(_synapse.s_in)
                _synapse.a_out.connect(port)
                self.out_conn.append(_synapse)
        else:
            if delays.shape != weights.shape:
                raise ValueError("Delays must have same shape as weights.")
            for idx in range(len(self.lif)):
                _synapse = DelaySparse(
                    weights=weights[:, idx][:, np.newaxis],
                    delays=delays[:, idx][:, np.newaxis]
                )
                self.lif[0].s_out.connect(_synapse.s_in)
                _synapse.a_out.connect(port)
                self.out_conn.append(_synapse)

    # def connect_from() -> None:
    #	"""Connect an input process to the inputs of the synapses"""
