"""
Spike Template Matching Network (STMNet)
========================================

Spiking neural network that recognizes spatio-temporal spike sequences using a
synapse delay encoding.
"""

import numpy as np
import scipy.sparse as sparse

import stmnet.proc as proc
import stmnet.configuration as cfg


class TemplateMatchingNetwork:
    def __init__(
        self,
        feature_detector_key: dict,
        templates: dict,
        binary_detector_key: dict,
        config: cfg.TemplateMatchingNetworkConfig,
    ):
        # setup keys dict
        self.keys = {}
        # add params dict
        # self.params = dict(**parameters)
        self.cfg = config

        # add input detector keys
        self.keys['feature_detector'] = [
            did for did in feature_detector_key.values()
        ]
        self.keys['binary_detector'] = [
            did for did in binary_detector_key.values()
        ]


        # add templates
        self.templates = dict(**templates)

        # get syllables from the templates
        self.keys['template'] = [tid for tid in templates.keys()]
        self.syllables = sorted(set(
            [tid.split('.')[0] for tid in self.keys['template']]
        ))

        self.n_feature_detectors = len(self.keys['feature_detector'])
        self.n_binary_detectors = len(self.keys['binary_detector'])
        self.n_templates = len(self.keys['template'])
        self.n_syllables = len(self.syllables)

        # create templates from the template dict
        self.template_arrs = self.make_templates()
        self.max_onset_delay = int(np.max(self.template_arrs['onset']))

    def trim_template_sequence_select_first(self, template_sequence):
        """Trim template sequence s.t. there is at most one spike per detector.
        For detectors with multiple spikes, select the first spike.
        """
        # sort spike sequence by spike time (second value in pair)
        template_sequence = sorted(template_sequence, key=lambda x: x[1])

        # remove any repeated detectors from the sequence
        used_idxs = []
        sequence = []
        for idx, time in template_sequence:
            if idx in used_idxs:
                continue

            used_idxs.append(idx)
            sequence.append((idx, time))

        return sequence

    def make_template_delay_arr(self, template_sequence, n_detectors):
        sequence = self.trim_template_sequence_select_first(template_sequence)

        # create spike time array
        spike_times = np.zeros(shape=(n_detectors,))
        for idx, time in sequence:
            spike_times[idx] = time

        # convert spike times to delays
        delays = np.zeros(shape=(n_detectors,))
        # leave dets w/out a spike time as delay 0
        delays[np.nonzero(spike_times)] = (
            np.max(spike_times)
            - spike_times[np.nonzero(spike_times)]
            + 1
        )

        return delays

    def make_templates(self):
        """Expects a nested dict template_sequences, keyed first by template id
        then by 'onset', 'offset'.

        Returns dict containing 'onset', 'offset', and 'key'
        """
        onset_delays = []
        offset_delays = []
        for tpt_id in self.keys['template']:
            tpt_seqs = self.templates[tpt_id]
            onset_delays.append(
                self.make_template_delay_arr(
                    tpt_seqs['onset'],
                    self.n_feature_detectors
                )
            )
            offset_delays.append(
                self.make_template_delay_arr(
                    tpt_seqs['offset'],
                    self.n_feature_detectors
                )
            )

        return {
            'onset': np.stack(onset_delays),
            'offset': np.stack(offset_delays)
        }


    def build_network(self):
        """Create the network layers and connections"""
        self.create_template_layers()
        self.create_template_prediction_layer()
        self.create_decision_layer()

    ###########################################################################
    # Creating Onset and Offset Network: Feature Detectors and Template Neurons
    def create_template_layers(self):
        self.FeatDet__Template_conn = {'Onset': {}, 'Offset': {}}
        self.TemplateLif = {'Onset': {}, 'Offset': {}}

        # Onset & Offset network build
        self._CreateOnsetTemplateNetwork()
        self._CreateOffsetTemplateNetwork()


    ###########################################################################
    # Creating Plateau Neuron Layer Connections
    def create_template_prediction_layer(self):
        self.Template__Plateau_conn = {'dendrite': {}, 'soma': {}}

        # connection mask for templates -> predictor
        connMask = np.array(
            [
                [
                    1 if tpt_id.split('.')[0] == syl
                    else 0 for tpt_id in self.keys['template']]
                for syl in self.syllables
            ]
        )

        # create the template -> predictor connections
        for tpt_idx in range(connMask.shape[1]):
            # Onset Template to Dendrite Connections
            self.Template__Plateau_conn['dendrite'][tpt_idx] = proc.DelaySparse(
                weights=sparse.csr_matrix(
                    self.cfg.synapse.tpt_tpd.weight_dend
                    * connMask[:, tpt_idx][:, np.newaxis]
                ),
                delays=sparse.csr_matrix(
                    0 * connMask[:, tpt_idx][:, np.newaxis]
                ),
                max_delay=self.max_onset_delay
            )

            # Offset Template to Soma Connections
            self.Template__Plateau_conn['soma'][tpt_idx] = proc.DelaySparse(
                weights=sparse.csr_matrix(
                    self.cfg.synapse.tpt_tpd.weight_soma
                    * connMask[:, tpt_idx][:, np.newaxis]
                ),
                delays=sparse.csr_matrix(
                    np.matrix(
                        int(self.max_onset_delay)
                        * connMask[:, tpt_idx][:, np.newaxis]
                    )
                ),
                max_delay=self.max_onset_delay
            )

        # vth_dend = int(templates_per_syllable * pct_dendrite_spikes / 5)
        vth_dend = int(
            self.n_templates
            / self.n_syllables
            * self.cfg.meta.pct_dendrite_spikes
            / 5
        )

        # Creating Plateau Neuron Layer
        self.Plateau_Neuron = proc.NxPlateau(
            shape=(self.n_syllables,), 
            dv_dend=self.cfg.neuron.template_prediction.dv_dend,
            dv_soma=self.cfg.neuron.template_prediction.dv_soma,
            vth_dend=vth_dend,
            vth_soma=self.cfg.neuron.template_prediction.vth_soma,
            name='template_prediction'
        )

        # connect onset template -> template prediction dendrite
        for neuron, synapse in zip(
            self.TemplateLif['Onset'].values(),
            self.Template__Plateau_conn['dendrite'].values()
        ):
            neuron.s_out.connect(synapse.s_in)
            synapse.a_out.connect(self.Plateau_Neuron.a_dend_in)

        # connect offset template -> template prediction soma
        for neuron, synapse in zip(
            self.TemplateLif['Offset'].values(),
            self.Template__Plateau_conn['soma'].values()
        ):
            neuron.s_out.connect(synapse.s_in)
            synapse.a_out.connect(self.Plateau_Neuron.a_soma_in)


    ###########################################################################
    # Creating Decision Layer
    def create_decision_layer(self):
        self.Plateau_to_Decision_Conn = proc.Sparse(
            weights=sparse.csr_matrix(2 * np.eye(self.n_syllables))
        )

        self.DecisionLayer = proc.LIFRefractory(
            name="Decision LIF",
            shape=(self.n_syllables,),
            du=self.cfg.neuron.decision.du,
            dv=self.cfg.neuron.decision.dv,
            vth=self.cfg.neuron.decision.vth,
            refractory_period=self.cfg.neuron.decision.refractory_period,
        )

        # Plateau Neuron's --> OutputLIF
        self.Plateau_Neuron.s_out.connect(self.Plateau_to_Decision_Conn.s_in)
        self.Plateau_to_Decision_Conn.a_out.connect(self.DecisionLayer.a_in)


    ###########################################################################
    # Template Layer Creation
    def create_template_lif_layer(self, mode:str):
        """Create template lif layer for specified mode ('onset'|'offset')."""

        # calculate vThMants for each template
        n_dets_per_tpt = np.sum(
            np.where(self.template_arrs[mode], 1, 0), axis=1
        )
        vThMants = np.array([
            int(np.max(
                [
                    1,
                    int(
                        self.cfg.meta.threshold_factor
                        * self.cfg.synapse.fin_tpt.weight
                        * n_dets
                    )
                ]
            ))
            for n_dets in n_dets_per_tpt
        ])

        # forces templates for a given syllable to have the same vThMant
        if self.cfg.meta.normalize_syllables:
            syl_map = np.array(
                [
                    self.keys['syllable'].index(syl)
                    for syl in self.keys['template']
                ]
            )
            for i in range(len(self.keys['syllable'])):
                vThMants[syl_map == i] = np.min(vThMants[syl_map == i])

        return {
            idx: proc.LIFRefractory(
                name=f"template_{mode}_{idx}",
                shape=(1,),
                vth=vThMant,
                du=self.cfg.neuron.template.du,
                dv=self.cfg.neuron.template.dv,
                refractory_period=self.cfg.neuron.template.refractory_period,
            )
            for idx, vThMant in enumerate(vThMants)
        }

    def create_feature_detector_template_connections(self, mode):
        """Create dict of DelaySparse processes for connecting input layer
        to the template layer specified by mode ('onset' | 'offset').
        
        Assumes that all inputs are from a single layer (feature and binary)
        """
        # check if padding needed for binary detector inputs
        # requires existing binary detectors and mode == 'onset'
        if len(self.keys['binary_detector']) > 0 and mode == 'onset':
            conn_map = np.concatenate(
                [
                    self.template_arrs[mode], 
                    np.zeros(
                        shape=(
                            self.template_arrs[mode].shape[0],
                            len(self.keys['binary_detector'])
                        )
                    )
                ],
                axis=1
            )
        else:
            conn_map = self.template_arrs[mode].copy()

        return {
            idx: proc.DelaySparse(
                weights=sparse.csr_matrix(
                    np.where(tpt_delays, self.cfg.synapse.fin_tpt.weight, 0)
                ),
                delays=sparse.csr_matrix(tpt_delays),
            )
            for idx, tpt_delays in enumerate(conn_map.astype(int))
        }

    def _CreateOnsetTemplateNetwork(self):
        """
        Creating Onset Template Network, with Feature Detectors 
        and Template Neurons (LIF)
        """
        self.TemplateLif['Onset'] = self.create_template_lif_layer('onset')

        self.FeatDet__Template_conn['Onset'] = (
            self.create_feature_detector_template_connections('onset')
        )

        for idx, conn in self.FeatDet__Template_conn['Onset'].items():
            conn.a_out.connect(self.TemplateLif['Onset'][idx].a_in)

        return self

    def _CreateOffsetTemplateNetwork(self):
        """
        Creating Offset Template Network, with Feature Detectors 
        and Template Neurons (LIF)
        """
        self.TemplateLif['Offset'] = self.create_template_lif_layer('offset')

        self.FeatDet__Template_conn['Offset'] = (
            self.create_feature_detector_template_connections('offset')
        )

        for idx, conn in self.FeatDet__Template_conn['Offset'].items():
            conn.a_out.connect(self.TemplateLif['Offset'][idx].a_in)

        return self

    def connect_input_layer(self, s_out, mode:str):
        """Connect the 'mode' templates to the input port s_out.

        Parameters
        ----------
        s_out: OutPort
            Spike out port of the input layer.
        mode: str ['onset', 'offset']
            Which template layer to connect to the input.
        """
        for conn in self.FeatDet__Template_conn[mode.capitalize()].values():
            s_out.connect(conn.s_in)

    def connect_output_layer(self, a_in):
        """Connect the decision layer to the output port a_in.

        Parameters
        ----------
        a_in: Input Port
            Input port to the spike receiver.
            Must have shape (self.n_syllables,)
        """
        out_conn = proc.Sparse(
            weights=sparse.csr_matrix(np.eye(self.n_syllables))
        )
        self.DecisionLayer.s_out.connect(out_conn.s_in)
        out_conn.a_out.connect(a_in)

    ###########################################################################
    # def create_spike_injector_network(
    #     self,
    #     onset_feature_spikes,
    #     offset_feature_spikes,
    #     binary_feature_spikes,
    # ):
    #     self.onset_feature_spikes = onset_feature_spikes
    #     self.offset_feature_spikes = offset_feature_spikes
    #     self.binary_feature_spikes = binary_feature_spikes

    #     """
    #     Generate the Spike Injectors and the PytoNx Adapters
    #     to create the input network and connections to 
    #     Template "Dense" connections.
    #     """
    #     #Onset Network
    #     # add binary detector spikes to onset feature spikes
    #     onset_input_spikes = np.concatenate(
    #         [onset_feature_spikes, binary_feature_spikes],
    #         axis=0
    #     )
    #     self.SpikeInjector['Onset'] = SpikeIn(data=onset_input_spikes.astype(int))
    #     self.PytoNx['Onset'] = PyToNxAdapter(shape=((onset_input_spikes.shape[0],)))
    #     self.NxtoDense['Onset'] = Dense(weights=np.eye(onset_input_spikes.shape[0])*2)
    #     self.DensetoLifNx['Onset'] = LIF(u=0,
    #                                      v=0,
    #                                      du=4095,
    #                                      dv=4095,
    #                                      bias_mant=0,
    #                                      bias_exp=0,
    #                                      vth=1,
    #                                      shape=(onset_input_spikes.shape[0],),
    #                                      name="lifnx_onset"
    #                                      )
    #     for num_syl in range(self.num_syllables):
    #         # onset weights and delays need zeros added to account for binary
    #         pad_zeros = np.zeros(
    #             shape=(
    #                 self.input_weights[num_syl].shape[0],
    #                 binary_feature_spikes.shape[0]
    #             ),
    #             dtype=int
    #         )
    #         self.FeatDet__Template_conn['Onset'][num_syl] = DelayDense(
    #             weights=np.concatenate(
    #                 [self.input_weights[num_syl], pad_zeros],
    #                 axis=1
    #             ).astype(int),
    #             delays=np.concatenate(
    #                 [self.delays_per_weight_InConn[num_syl], pad_zeros],
    #                 axis=1
    #             ).astype(int),
    #             max_delay=self._find_max_delay(self.delays_per_weight_InConn)
    #         )
    #         self.FeatDet__Template_conn['Onset'][num_syl].a_out.connect(
    #             self.TemplateLif['Onset'][num_syl].a_in
    #         )


    #     #Offset Network
    #     self.SpikeInjector['Offset'] = SpikeIn(data=self.offset_feature_spikes.astype(int))
    #     self.PytoNx['Offset'] = PyToNxAdapter(shape=((self.offset_feature_spikes.shape[0],)))
    #     self.NxtoDense['Offset'] = Dense(weights=np.eye(self.offset_feature_spikes.shape[0])*2)
    #     self.DensetoLifNx['Offset'] = LIF(u=0,
    #                                       v=0,
    #                                       du=4095,
    #                                       dv=4095,
    #                                       bias_mant=0,
    #                                       bias_exp=0,
    #                                       vth=1,
    #                                       shape=(self.offset_feature_spikes.shape[0],),
    #                                       name="lifnx_offset"
    #                                       )
    #     for num_syl in range(self.num_syllables):
    #         self.FeatDet__Template_conn['Offset'][num_syl] = DelayDense(
    #             weights=self.input_weights[num_syl],
    #             delays=self.delays_per_weight_InConn[num_syl],
    #             max_delay=self._find_max_delay(self.delays_per_weight_InConn),
    #             log_config=LogConfig(level=logging.INFO)
    #         )
    #         self.FeatDet__Template_conn['Offset'][num_syl].a_out.connect(
    #             self.TemplateLif['Offset'][num_syl].a_in
    #         )


    #     # Creating SpikeInput --> PyToNxAdapter connections
    #     self.SpikeInjector['Onset'].s_out.connect(self.PytoNx['Onset'].inp)
    #     self.PytoNx['Onset'].out.connect(self.NxtoDense['Onset'].s_in)
    #     self.NxtoDense['Onset'].a_out.connect(self.DensetoLifNx['Onset'].a_in)

    #     self.SpikeInjector['Offset'].s_out.connect(self.PytoNx['Offset'].inp)
    #     self.PytoNx['Offset'].out.connect(self.NxtoDense['Offset'].s_in)
    #     self.NxtoDense['Offset'].a_out.connect(self.DensetoLifNx['Offset'].a_in)

    #     for num_syl in range(self.num_syllables):
    #         self.DensetoLifNx['Onset'].s_out.connect(self.FeatDet__Template_conn['Onset'][num_syl].s_in)
    #         self.DensetoLifNx['Offset'].s_out.connect(self.FeatDet__Template_conn['Offset'][num_syl].s_in)

    #     return self

    # def create_spike_receiver(self, num_timesteps):
    #     self.num_timesteps = num_timesteps
    #     # Creating Sink to read data
    #     self.DecisionLIF_SinkReadout = Dense(weights=np.eye(self.num_syllables))
    #     self.SpikeOutNxtoPy = NxToPyAdapter(shape=(self.num_syllables,))
    #     self.SpikeSink = Sink(shape=(self.num_syllables,), buffer=num_timesteps) 

    #     self.DecisionLayer.s_out.connect(self.DecisionLIF_SinkReadout.s_in)
    #     self.DecisionLIF_SinkReadout.s_in.connect(self.SpikeOutNxtoPy.inp)
    #     self.SpikeOutNxtoPy.out.connect(self.SpikeSink.a_in)


class BinaryPredictionNetwork:
    """Binary prediction components of the template matching network.

    Layers
    ------
    pre_coincidence
    coincidence
    binary_prediction
    binary_features (* not made in setup)

    Connections
    -----------
    pre_coincidence_to_coincidence
    coincidence_to_binary_prediction
    binary_prediction_to_self

    External Connections
    (*not made in setup)
    --------------------
    template_onset_to_pre_coincidence
    coincidence_to_template_prediction
    binary_feature_to_binary_prediction
    binary_prediction_to_decision


    Parameters
    ----------
    pre_coincidence_dv
    pre_coincidence_vth
    pre_coincidence_refractory_period
    coincidence_dv
    coincidence_vth
    binary_prediction_dv_soma
    binary_prediction_dv_dend
    binary_prediction_vth_soma
    binary_prediction_vth_dend
    pre_coincidence_to_coincidence_weight
    coincidence_to_binary_prediction_weight
    binary_prediction_to_self_weight

    External Parameters
    -------------------
    templates_per_syllable
    template_onset_to_pre_coincidence_weight
    template_onset_to_pre_coincidence_delay
    coincidence_to_template_prediction_weight
    binary_feature_to_binary_prediction_weight
    binary_feature_to_binary_prediction_delay
    binary_prediction_to_decision_weight

    Layer Keys
    ----------
    input
    template
    binary
    syllable
    binary_syllable
    coincidence
    binary_pair
    """
    def __init__(
        self,
        template_key: list,
        binary_detector_key: list,
        config: cfg.TemplateMatchingNetworkConfig,
        max_onset_delay: int,
    ):
        # init layers
        self.pre_coincidence = None
        self.coincidence = None
        self.binary_prediction = None
        self.binary_features = None
        # init connections
        self.pre_coincidence_to_coincidence = None
        self.coincidence_to_binary_prediction = None
        self.binary_prediction_to_self = None
        # init external connections
        self.onset_template_to_pre_coincidence = None
        self.coincidence_to_template_prediction = None
        self.binary_feature_to_binary_prediction = None
        self.binary_prediction_to_decision = None

        self.cfg = config
        self.max_onset_delay = max_onset_delay

        # layer keys
        self.layer_keys = {
            'template': template_key,
            'binary': binary_detector_key,
            'syllable': None,
            'binary_syllable': None,
            'coincidence': None,
            'binary_pair': None,
            'input': None,
        }
        # self._set_parameters(parameters)
        self._create_layer_keys()

        # get n_templates and n_syllables
        self.n_templates = len(self.layer_keys['template'])
        self.n_syllables = len(self.layer_keys['syllable'])

    def build_network(self):
        """Build the Binary Prediction Network."""
        self._build_pre_coincidence_layer()
        self._build_coincidence_layer()
        self._connect_pre_coincidence_to_coincidence()
        self._build_binary_prediction_layer()
        self._connect_coincidence_to_binary_prediction()
        self._connect_binary_prediction_reset()


    # def _set_parameters(self, net_parameters):
    #     """Set the class's network parameters and external parameters, using
    #     the values from the supplied dict.
        
    #     Assumes all required parameters are present.
    #     """
    #     # parameters
    #     for param in self.params.keys():
    #         if param in net_parameters:
    #             self.params[param] = net_parameters[param]
    #         else:
    #             raise KeyError(f"Parameter '{param}' not supplied!")
    #     # external parameters
    #     for param in self.external_params.keys():
    #         if param in net_parameters:
    #             self.external_params[param] = net_parameters[param]
    #         else:
    #             raise KeyError(f"External parameter '{param}' not supplied!")


    def _create_layer_keys(self):
        """Create inferred layer keys and populate class's layer_keys dict.
        Does not create input key.
        """
        # syllable-only keys for templates and binary detectors
        template_syl_key = [
            tid.split('.')[0] for tid in self.layer_keys['template']
        ]
        binary_syl_key = [
            bid[0:3] for bid in self.layer_keys['binary']
        ]

        self.layer_keys['template_syllable'] = template_syl_key
        self.layer_keys['binary_detector_syllable'] = binary_syl_key
        # generated keys
        self.layer_keys['syllable'] = []
        for syl in template_syl_key:
            if syl not in self.layer_keys['syllable']:
                self.layer_keys['syllable'].append(syl)

        self.layer_keys['binary_syllable'] = []
        for syl_pair in binary_syl_key:
            for syl in syl_pair.split('.'):
                if syl not in self.layer_keys['binary_syllable']:
                    self.layer_keys['binary_syllable'].append(syl)

        self.layer_keys['coincidence'] = []
        for syl_pair in binary_syl_key:
            coi_pair = '^'.join(sorted(syl_pair.split('.')))
            if coi_pair not in self.layer_keys['coincidence']:
                self.layer_keys['coincidence'].append(coi_pair)

        self.layer_keys['binary_pair'] = []
        for syl_pair in binary_syl_key:
            if syl_pair not in self.layer_keys['binary_pair']:
                self.layer_keys['binary_pair'].append(syl_pair)

    def _build_pre_coincidence_layer(self):
        # same vth as template prediction dendrites
        # vth = int(templates_per_syllable * pct_dendrite_spikes / 5)
        vth = int(
            self.n_templates
            / self.n_syllables
            * self.cfg.meta.pct_dendrite_spikes
            / 5
        )

        # network layers
        self.pre_coincidence = proc.LIFRefractory(
            shape=(len(self.layer_keys['binary_syllable']),),
            vth=vth,
            du=self.cfg.neuron.pre_coincidence.du,
            dv=self.cfg.neuron.pre_coincidence.dv,
            refractory_period=self.cfg.neuron.pre_coincidence.refractory_period,
        )

    def _build_template_to_pre_coincidence_connections(self):
        _connmask = np.array([
            [
                1 if in_syl == out_syl else 0
                for in_syl in self.layer_keys['template_syllable']
            ] for out_syl in self.layer_keys['binary_syllable']
        ])
        self.onset_template_to_pre_coincidence = {}
        for idx, conn_col in enumerate(_connmask.T):
            # skip templates which don't connect to any coincidence neurons
            if np.sum(conn_col) == 0:
                continue

            self.onset_template_to_pre_coincidence[idx] = proc.DelaySparse(
                weights=sparse.csr_matrix(
                    self.cfg.synapse.tpt_pcd.weight
                    * conn_col[:, np.newaxis]
                ),
                delays=sparse.csr_matrix(
                    self.cfg.synapse.tpt_pcd.delay
                    * conn_col[:, np.newaxis]
                )
            )

        for conn in self.onset_template_to_pre_coincidence.values():
            conn.a_out.connect(self.pre_coincidence.a_in)

    def _build_coincidence_layer(self):
        self.coincidence = proc.LIF(
            shape=(len(self.layer_keys['coincidence']),),
            du=self.cfg.neuron.coincidence.du,
            dv=self.cfg.neuron.coincidence.dv,
            vth=self.cfg.neuron.coincidence.vth,
        )

    def _connect_pre_coincidence_to_coincidence(self):
        _connmask = np.array([
            [
                1 if in_syl in out_syls.split('^') else 0
                for in_syl in self.layer_keys['binary_syllable']
            ] for out_syls in self.layer_keys['coincidence']
        ])
        self.pre_coincidence_to_coincidence = proc.Sparse(
            weights=self.cfg.synapse.pcd_ccd.weight * _connmask
        )
        try:
            self.pre_coincidence.s_out.connect(
                self.pre_coincidence_to_coincidence.s_in
            )
        except:
            print("No Pre-Coincidence layer found!")
        try:
            self.pre_coincidence_to_coincidence.a_out.connect(
                self.coincidence.a_in
            )
        except:
            print("No Coincidence layer found!")

    def _build_binary_prediction_layer(self):
        self.binary_prediction = proc.NxPlateau(
            name='binary_prediction',
            shape=(len(self.layer_keys['binary_pair']),),
            dv_soma=self.cfg.neuron.binary_prediction.dv_soma,
            dv_dend=self.cfg.neuron.binary_prediction.dv_dend,
            vth_soma=self.cfg.neuron.binary_prediction.vth_soma,
            vth_dend=self.cfg.neuron.binary_prediction.vth_dend,
        )

    def _connect_coincidence_to_binary_prediction(self):
        _connmask = np.array([
            [
                1 if (
                    coi_pair.replace('^','.') == bin_pair
                    or coi_pair.replace('^', '.')[::-1] == bin_pair
                ) else 0
                for coi_pair in self.layer_keys['coincidence']
            ] for bin_pair in self.layer_keys['binary_pair']
        ])
        self.coincidence_to_binary_prediction = proc.DelaySparse(
            weights=sparse.csr_matrix(
                self.cfg.synapse.ccd_bpd.weight * _connmask
            ),
            delays=sparse.csr_matrix(np.zeros(_connmask.shape, dtype=int)),
            max_delay=int(self.max_onset_delay)
        )

        self.coincidence.s_out.connect(
            self.coincidence_to_binary_prediction.s_in
        )
        self.coincidence_to_binary_prediction.a_out.connect(
            self.binary_prediction.a_dend_in
        )

    def _connect_binary_prediction_reset(self):
        _connmask = np.array([
            [
                1 if sorted(pair_a) == sorted(pair_b) else 0
                for pair_a in self.layer_keys['binary_pair']
            ] for pair_b in self.layer_keys['binary_pair']
        ])
        self.binary_prediction_to_self = proc.DelaySparse(
            # weight is inverse of coincidence -> binary prediction weight
            weights=sparse.csr_matrix(
                - self.cfg.synapse.ccd_bpd.weight * _connmask
            ),
            delays=sparse.csr_matrix(
                np.zeros(shape=_connmask.shape, dtype=int)
            ),
            max_delay=int(self.max_onset_delay)
        )
        self.binary_prediction.s_out.connect(
            self.binary_prediction_to_self.s_in
        )
        self.binary_prediction_to_self.a_out.connect(
            self.binary_prediction.a_dend_in
        )

    def _build_coincidence_to_template_prediction_connection(self):
        # weight = - templates_per_syllable
        weight = - int(self.n_templates / self.n_syllables)
        # connect coincidence layer to template prediction dendrites
        _connmask = np.array([
            [
                1 if out_syl in in_syls.split('^') else 0
                for in_syls in self.layer_keys['coincidence']
            ] for out_syl in self.layer_keys['syllable']
        ])
        self.coincidence_to_template_prediction = proc.DelaySparse(
            weights=sparse.csr_matrix(weight * _connmask),
            delays=sparse.csr_matrix(np.zeros(_connmask.shape, dtype=int)),
            max_delay=int(self.max_onset_delay)
        )
        self.coincidence.s_out.connect(
            self.coincidence_to_template_prediction.s_in
        )

    def _build_binary_feature_to_binary_prediction_connection(self):
        # connect binary features to binary prediction somas
        _connmask = np.array([
            [
                1 if '.'.join(det_pair.split('.')[0:2]) == pred_pair else 0
                for det_pair in self.layer_keys['input']
            ]
            for pred_pair in self.layer_keys['binary_pair']
        ])
        self.binary_feature_to_binary_prediction = proc.DelaySparse(
            weights=sparse.csr_matrix(
                self.cfg.synapse.bin_bpd.weight * _connmask
            ),
            delays=sparse.csr_matrix(self.max_onset_delay * _connmask),
            max_delay=int(self.max_onset_delay)
        )
        self.binary_feature_to_binary_prediction.a_out.connect(
            self.binary_prediction.a_soma_in
        )

    def _build_binary_prediction_to_decision_connection(self):
        # connect binary prediction to decision neurons
        _connmask = np.array([
            [
                1 if dec_syl == pred_pair[0] else 0
                for pred_pair in self.layer_keys['binary_pair']
            ] for dec_syl in self.layer_keys['syllable']
        ])
        self.binary_prediction_to_decision = proc.Sparse(
            weights=self.cfg.synapse.bpd_dec.weight * _connmask
        )
        self.binary_prediction.s_out.connect(
            self.binary_prediction_to_decision.s_in
        )

    def _connect_to_template_layer(self, template_layers):
        """Connect to the onset template layers of the template matching
        network.
        """
        for idx, conn in self.onset_template_to_pre_coincidence.items():
            template_layers[idx].s_out.connect(conn.s_in)

    def _connect_to_template_prediction_layer(self, template_prediction):
        """Connect to the template prediction layer of the template matching
        network.
        """
        self.coincidence_to_template_prediction.a_out.connect(
            template_prediction.a_dend_in
        )

    def _connect_to_decision_layer(self, decision):
        """Connect to the decision layer of the template matching network.
        """
        self.binary_prediction_to_decision.a_out.connect(decision.a_in)

    def _connect_to_binary_feature_layer(self, binary_feature):
        """Connect the binary detector inputs to the binary prediction somas.
        """
        binary_feature.s_out.connect(
            self.binary_feature_to_binary_prediction.s_in
        )

    def connect_to_template_matching_network(self, tmn):
        """Connect to Template Matching Network stored in the tmn class.
        """
        self._build_template_to_pre_coincidence_connections()
        self._connect_to_template_layer(tmn.TemplateLif['Onset'])
        self._build_coincidence_to_template_prediction_connection()
        self._connect_to_template_prediction_layer(tmn.Plateau_Neuron)
        self._build_binary_prediction_to_decision_connection()
        self._connect_to_decision_layer(tmn.DecisionLayer)

    def connect_to_input_layer(self, feature_inp, input_key:list):
        self.layer_keys['input'] = input_key
        self._build_binary_feature_to_binary_prediction_connection()
        self._connect_to_binary_feature_layer(feature_inp['onset'])

class TMNIO:
    def __init__(self,):
        
        self.Lif_On = None
        self.Plateau_Neuron = None
        self.DecisionLayer = None
        self.spike_data = {}

    @classmethod
    def create_tmn_io_processes(
        cls, template_input_spikes, layer_keys, runtime
    ):
        """Create io processes with the same format as the TMN. The processes
        are stored in a class that has similar structure as the required pieces
        of the TMN.
        """
        template_input_data = np.zeros(
            shape=(len(layer_keys['template']), runtime)
        )
        for (i, t) in template_input_spikes:
            template_input_data[i, t] = 1

        # split templates by syllable
        split_idxs = [
            idx for idx in range(1, len(layer_keys['template']))
            if layer_keys['template'][idx] != layer_keys['template'][idx - 1]
        ]

        template_input_data = np.split(template_input_data, split_idxs, axis=0)
        template_sgs = {
            idx: proc.SpikeIn(data=template_input_data[idx])
            for idx in range(len(layer_keys['syllable']))
        }

        # decision output
        decision_sr = proc.SpikeOut(
            shape=(len(layer_keys['syllable']),),
            buffer=runtime
        )

        # Plateau Neuron spike receiver
        plateau_sr = proc.SpikeOut(
            shape=(len(layer_keys['syllable']),),
            buffer=runtime
        )
        plateau_sr.__setattr__('a_dend_in', plateau_sr.a_in)

        obj = cls()
        obj.Lif_On = template_sgs
        obj.Plateau_Neuron = plateau_sr
        obj.DecisionLayer = decision_sr
        return obj

    def get_data(self):
        self.spike_data['template_onset'] = {
            idx: sg.data.get() for idx, sg in self.Lif_On.items()
        }
        self.spike_data['template_prediction'] = self.Plateau_Neuron.data.get()
        self.spike_data['decision'] = self.DecisionLayer.data.get()


def create_layer_keys(template_key, binary_key):
    layer_keys = {
        'template': template_key,
        'binary': binary_key
    }
    # generated keys
    layer_keys['syllable'] = []
    for syl in layer_keys['template']:
        if syl not in layer_keys['syllable']:
            layer_keys['syllable'].append(syl)

    layer_keys['binary_syllable'] = []
    for syl_pair in layer_keys['binary']:
        for syl in syl_pair.split('.'):
            if syl not in layer_keys['binary_syllable']:
                layer_keys['binary_syllable'].append(syl)

    layer_keys['coincidence'] = []
    for syl_pair in layer_keys['binary']:
        coi_pair = '^'.join(sorted(syl_pair.split('.')))
        if coi_pair not in layer_keys['coincidence']:
            layer_keys['coincidence'].append(coi_pair)

    layer_keys['binary_pair'] = []
    for syl_pair in layer_keys['binary']:
        if syl_pair not in layer_keys['binary_pair']:
            layer_keys['binary_pair'].append(syl_pair)

    return layer_keys


class NetworkDescription:
    """Defines architecture of a STMNet"""
    def __init__(self, feature_detectors, binary_detectors, templates):
        self.feature_ids: dict
        self.template_ids: dict
        self.binary_ids: dict
        self.keys: dict
        self.templates: dict

        self._data: dict = {
            'feature_detectors' : feature_detectors,
            'binary_detectors' : binary_detectors,
            'templates' : templates,
            'keys': {},
        }

        self._build()

    def __getitem__(self, arg):
        if isinstance(arg, str) and arg.endswith('_key'):
            return self._data['keys'][arg.replace('_key', '')]
        return getattr(self, arg)

    def __setitem__(self, arg, value):
        setattr(self, arg, value)

    def _build(self, split_char:str='.'):
        """Uses the input detector and template information to complete the
        network description.
        """
        self._data['template_ids'] = {
            idx: tpt_id
            for idx, tpt_id in enumerate(self.templates.keys())
        }

        # layer keys, shorthand forms of ids
        keys = self._data['keys']

        keys['feature_detector'] = [
            self.feature_detectors[idx].split(split_char)[0]
            for idx in range(len(self._data['feature_detectors']))
        ]

        keys['binary_detector'] = [
            split_char.join(self.binary_detectors[idx].split(split_char)[0:2])
            for idx in range(len(self._data['binary_detectors']))
        ]

        keys['template'] = [
            self._data['template_ids'][idx].split(split_char)[0]
            for idx in range(len(self._data['template_ids']))
        ]

        keys['syllable'] = []
        for syl in keys['template']:
            if syl not in keys['syllable']:
                keys['syllable'].append(syl)

        keys['coincidence'] = []
        for bin_pair in keys['binary_detector']:
            _sort_pair = '|'.join(sorted(bin_pair.split(split_char)))
            if _sort_pair not in keys['coincidence']:
                keys['coincidence'].append(_sort_pair)
        
        keys['binary_pair'] = []
        for bin_pair in keys['coincidence']:
            keys['binary_pair'].extend(
                [bin_pair, '.'.join(bin_pair.split('|')[::-1])]
            )

    @property
    def feature_ids(self):
        """Get the feature detectors for the network."""
        return self._data['feature_detectors']

    @property
    def template_ids(self):
        """Get the templates ids for the network."""
        return self._data['template_ids']

    @property
    def binary_ids(self):
        """Get the binary detectors for the network."""
        return self._data['binary_detectors']

    @property
    def keys(self):
        """Get the keys for the network."""
        return self._data['keys']

    @property
    def templates(self):
        """Get the templates for the network."""
        return self._data['templates']

    @property
    def feature_detectors(self):
        """[HERE FOR BACKWARDS COMPATIBILITY]
        Get the feature detectors for the network.
        """
        return self._data['feature_detectors']

    @property
    def binary_detectors(self):
        """[HERE FOR BACKWARDS COMPATIBILITY]
        Get the binary detectors for the network.
        """
        return self._data['binary_detectors']

    @property
    def template_key(self):
        """[HERE FOR BACKWARDS COMPATIBILITY]
        Get the template key for the network.
        """
        return self._data['template_ids']

    def print(self):
        print("Network Description")
        print("-------------------")
        print("Feature Detectors:")
        for idx, det in self._data['feature_detectors'].items():
            print(f"  {idx:>2}: {det}")
        print("Binary Detectors:")
        for idx, det in self._data['binary_detectors'].items():
            print(f"  {idx:>2}: {det}")
        print("Templates:")
        for tid, tdata in self._data['templates'].items():
            print(f"  {tid}:")
            print(f"     Onset: {', '.join(str(p) for p in tdata['onset'])}")
            print(f"    Offset: {', '.join(str(p) for p in tdata['offset'])}")

    def _trim_template_sequence_select_first(self, template_sequence):
        """Trim template sequence s.t. there is at most one spike per detector.
        For detectors with multiple spikes, select the first spike.
        """
        # sort spike sequence by spike time (second value in pair)
        template_sequence = sorted(template_sequence, key=lambda x: x[1])

        # remove any repeated detectors from the sequence
        used_idxs = []
        sequence = []
        for idx, time in template_sequence:
            if idx in used_idxs:
                continue

            used_idxs.append(idx)
            sequence.append((idx, time))

        return sequence

    def make_template_maps(self):
        """
        Create the masked arrays encoding the templates as delays. Returns a
        dict containing 'onset' and 'offset' keys, with the corresponding
        arrays as the values.
        """
        # trim and order the template spike sequences
        template_sequences = {
            onoff: [
                self._trim_template_sequence_select_first(
                    self['templates'][self._data['template_ids'][idx]][onoff]
                )
                for idx in range(len(self._data['template_ids']))
            ]
            for onoff in ['onset', 'offset']
        }

        # make the template map spike sequence arrays
        template_map = {}
        for onoff, sequences in template_sequences.items():
            spike_times = np.zeros(
                shape=(
                    len(sequences),
                    len(self._data['feature_detectors'])
                )
            )
            for tpt_idx, sequence in enumerate(sequences):
                for det_idx, time in sequence:
                    spike_times[tpt_idx, det_idx] = time
            template_map[onoff] = np.ma.masked_equal(
                spike_times.astype(int).copy(),
                0
            )

        # convert spike sequence times to delays (relative to template end)
        for onoff, spike_times in template_map.items():
            template_map[onoff] = np.ma.array(
                np.max(spike_times, axis=1, keepdims=True) - spike_times,
                fill_value=0,
                dtype=int
            )

        # shift onset templates so they are relative to max onset delay
        template_map['onset'] = (
            template_map['onset']
            + np.max(template_map['onset'])
            - np.max(template_map['onset'], axis=1, keepdims=True)
        )

        return template_map
