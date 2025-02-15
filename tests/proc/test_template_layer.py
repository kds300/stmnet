"""Unit Tests for Template Layer

Tests for the non-Lava template layer class.

"""

import unittest

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

import stmnet.proc as proc
from stmnet.proc.template.template_layer import TemplateLayer


class TestTemplateLayer(unittest.TestCase):
    """Unit tests for the TemplateLayer class.
    """
    def test_default_init(self):
        """Test default initialization of TemplateLayer objects."""
        delays = np.array(
            [
                [1, 2, 3, 4, 0, 0],
                [4, 3, 2, 1, 0, 0],
                [0, 0, 0, 0, 1, 3],
            ]
        )
        default_weight = 1
        default_f = 0.99

        default_vth = np.array(
            [
                int(np.max(
                    [
                        1,
                        default_f * default_weight * n_dets
                    ]
                ))
                for n_dets in [4, 4, 2]
            ]
        )


        template = TemplateLayer(delays=delays)

        for idx, _sparse in enumerate(template.sparse):
            self.assertTrue(
                np.array_equal(
                    delays[idx, :][np.newaxis, :],
                    _sparse.delays.get()
                )
            )

            self.assertTrue(
                np.array_equal(
                    np.where(delays[idx, :][np.newaxis, :], default_weight, 0),
                    _sparse.weights.get().toarray()
                )
            )

        for idx, _lif in enumerate(template.lif):
            self.assertEqual(default_vth[idx], _lif.vth.get())

    def test_custom_init(self):
        """Test custom initialization of TemplateLayer objects."""
        delays = np.array(
            [
                [1, 2, 3, 4, 0, 0],
                [4, 3, 2, 1, 0, 0],
                [0, 0, 0, 0, 1, 3],
            ]
        )
        weight = 2
        f = 0.4
        du = 4095
        dv = 1000

        vth = np.array(
            [
                int(np.max(
                    [
                        1,
                        f * weight * n_dets
                    ]
                ))
                for n_dets in [4, 4, 2]
            ]
        )


        template = TemplateLayer(
            delays=delays, f=f, weights=weight, du=du, dv=dv
        )

        for idx, _sparse in enumerate(template.sparse):
            self.assertTrue(
                np.array_equal(
                    delays[idx, :][np.newaxis, :],
                    _sparse.delays.get()
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.where(delays[idx, :][np.newaxis, :], weight, 0),
                    _sparse.weights.get().toarray()
                )
            )

        for idx, _lif in enumerate(template.lif):
            self.assertEqual(vth[idx], _lif.vth.get())
            self.assertEqual(du, _lif.du.get())
            self.assertEqual(dv, _lif.dv.get())

    def test_cpu_execution(self):
        """Test execution using a single-sparse, single-lif template layer."""
        n_nrns = 1
        n_steps = 10
        inp_data = np.zeros(shape=(n_nrns, n_steps), dtype=int)
        # send spike at t=5
        inp_data[0, 5] = 1

        delays = np.eye(n_nrns, dtype=int)
        weights = 20
        f = 0.99
        du = 4095
        dv = 4095

        source = proc.SpikeIn(data=inp_data)
        layer = TemplateLayer(
            delays=delays,
            f=f,
            weights=weights,
            du=du,
            dv=dv
        )
        sink = proc.SpikeOut(shape=(n_nrns,), buffer=n_steps)
        sink_sparse = proc.Sparse(weights=np.eye(n_nrns))

        source.s_out.connect(layer.s_in)
        layer.s_out[0].connect(sink_sparse.s_in)
        sink_sparse.a_out.connect(sink.a_in)

        layer.run(
            condition=RunSteps(num_steps=n_steps),
            run_cfg=Loihi2SimCfg(select_tag='fixed_pt')
        )

        out_indices, out_times = np.nonzero(sink.data.get())

        layer.stop()

        self.assertTrue(np.array_equal(out_indices, np.array([0])))
        self.assertTrue(
            np.array_equal(out_times, np.array([8])),
            msg=f"{out_times=}"
        )

    def test_basic_templates(self):
        """Test running a network with actual templates."""
        delays = np.array(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 1, 1, 1]
            ]
        )

        inp_data = np.concatenate(
            [
                # match tpt 1
                np.eye(4)[:, ::-1],
                # match tpt 2
                np.eye(4),
                # match tpt 3
                np.zeros(shape=(4, 3)),
                np.ones(shape=(4, 1)),
                # buffer for processing
                np.zeros(shape=(4, 10))
            ],
            axis=1
        )

        source = proc.SpikeIn(data=inp_data)
        template = TemplateLayer(delays=delays)
        sink = proc.SpikeOut(shape=(3,), buffer=inp_data.shape[1])

        source.s_out.connect(template.s_in)
        for i in range(3):
            _weights = np.zeros(shape=(3, 1))
            _weights[i, 0] = 1
            _sparse = proc.Sparse(weights=_weights)
            template.lif[i].s_out.connect(_sparse.s_in)
            _sparse.a_out.connect(sink.a_in)

        template.run(
            condition=RunSteps(num_steps=inp_data.shape[1]),
            run_cfg=Loihi2SimCfg(select_tag='fixed_pt')
        )

        out_data = sink.data.get()

        template.stop()

        out_spikes = np.stack(np.nonzero(out_data))

        self.assertTrue(
            np.array_equal(
                out_spikes,
                np.array([[0, 1, 2], [6, 10, 14]])
            ),
            msg=f"{out_spikes=}"
        )

    def test_connect_sparse(self):
        """Test connecting LIF neurons to a receiving process."""
        delays = np.array(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 1, 1, 1]
            ]
        )

        inp_data = np.concatenate(
            [
                # match tpt 1
                np.eye(4)[:, ::-1],
                # match tpt 2
                np.eye(4),
                # match tpt 3
                np.zeros(shape=(4, 3)),
                np.ones(shape=(4, 1)),
                # buffer for processing
                np.zeros(shape=(4, 10))
            ],
            axis=1
        )

        source = proc.SpikeIn(data=inp_data)
        template = TemplateLayer(delays=delays)
        sink = proc.SpikeOut(shape=(3,), buffer=inp_data.shape[1])

        source.s_out.connect(template.s_in)
        template.connect_sparse(sink.a_in, weights=np.eye(template.shape[1]))

        template.run(
            condition=RunSteps(num_steps=inp_data.shape[1]),
            run_cfg=Loihi2SimCfg(select_tag='fixed_pt')
        )

        out_data = sink.data.get()

        template.stop()

        out_spikes = np.stack(np.nonzero(out_data))

        self.assertTrue(
            np.array_equal(
                out_spikes,
                np.array([[0, 1, 2], [6, 10, 14]])
            ),
            msg=f"{out_spikes=}"
        )

if __name__ == "__main__":
    unittest.main()
