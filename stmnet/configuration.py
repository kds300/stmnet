"""Configuration for the STMNet.

The module contains many classes, mostly used for defining the configuration
structure for the STMNet. Most uses will only require 
`TemplateMatchingNetworkConfig`,
`load_config_data()`,
and `default_tmn_config()`.
"""

import os
import importlib
if importlib.util.find_spec('toml'):
    import toml
    config_loader = toml.load
    CONFIG_EXT = 'toml'
else:
    import json
    config_loader = json.load
    CONFIG_EXT = 'json'


DEFAULT_TMN_CONFIG_FILE = "default_tmn_config"


class Config:
    """Access configs as nested class attributes.

    Recursively converts nested dictionaries into nested Config objects.

    Original code at
    `kds300/junk_drawer<https://github.com/kds300/junk_drawer>`_
    """
    def __init__(self, parameters:dict={}, **kwargs):
        """
        Parameters
        ----------
        parameters: dict
            Nested dict containing the config parameters

        **kwargs
            Config parameters can also be specified using kwargs.
            If a parameter is specified both in parameters and as a kwarg, the
            kwarg will be used.
        """
        _params = {}
        for param_src in [parameters, kwargs]:
            for key, val in param_src.items():
                if type(val) == dict:
                    _params[key] = Config(val)
                else:
                    _params[key] = val
        self.__dict__.update(_params)

    def as_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.as_dict()
            else:
                out[k] = v
        return out

    def keys(self):
        return self.__dict__.keys()

    def __repr__(self):
        return f"{type(self).__name__}({self.as_dict()})"

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Config(value)
        self.__dict__[key] = value

    def __setattr__(self, name, value):
        self.__setitem__(name, value)


class TemplateNeuronConfig(Config):
    du: int
    dv: int
    refractory_period: int


class TemplatePredictionNeuronConfig(Config):
    du_dend: int
    dv_dend: int
    vth_soma: int
    du_soma: int
    dv_soma: int


class DecisionNeuronConfig(Config):
    vth: int
    du: int
    dv: int
    refractory_period: int


class PreCoincidenceNeuronConfig(Config):
    du: int
    dv: int
    refractory_period: int


class CoincidenceNeuronConfig(Config):
    vth: int
    du: int
    dv: int


class BinaryPredictionNeuronConfig(Config):
    dv_dend: int
    vth_dend: int
    dv_soma: int
    vth_soma: int


class FeatureInputToTemplateSynapseConfig(Config):
    weight: int


class TemplateToTemplatePredictionSynapseConfig(Config):
    weight_dend: int
    weight_soma: int


class TemplateToPreCoincidenceSynapseConfig(Config):
    weight: int
    delay: int


class PreCoincidenceToCoincidenceSynapseConfig(Config):
    weight: int


class CoincidenceToBinaryPredictionSynapseConfig(Config):
    weight: int


class BinaryInputToBinaryPredictionSynapseConfig(Config):
    weight: int


class BinaryPredictionToDecisionSynapseConfig(Config):
    weight: int


class MetaConfig(Config):
    threshold_factor: float
    pct_dendrite_spikes: float
    normalize_syllables: bool
    runtime_buffer: int
    loihi_frate: float


class NeuronConfig(Config):
    template: TemplateNeuronConfig
    template_prediction: TemplatePredictionNeuronConfig
    decision: DecisionNeuronConfig
    pre_coincidence: PreCoincidenceNeuronConfig
    coincidence: CoincidenceNeuronConfig
    binary_prediction: BinaryPredictionNeuronConfig


class SynapseConfig(Config):
    fin_tpt: FeatureInputToTemplateSynapseConfig
    tpt_tpd: TemplateToTemplatePredictionSynapseConfig
    tpt_pcd: TemplateToPreCoincidenceSynapseConfig
    pcd_ccd: PreCoincidenceToCoincidenceSynapseConfig
    ccd_bpd: CoincidenceToBinaryPredictionSynapseConfig
    bin_bpd: BinaryInputToBinaryPredictionSynapseConfig
    bpd_dec: BinaryPredictionToDecisionSynapseConfig


class TemplateMatchingNetworkConfig(Config):
    meta: MetaConfig
    neuron: NeuronConfig
    synapse: SynapseConfig

    @classmethod
    def load(cls, file_path:str):
        return cls(load_config_data(file_path))


def load_config_data(file_path:str) -> Config:
    with open(file_path, 'r') as f:
        _data = config_loader(f)
    return _data


def default_tmn_config() -> TemplateMatchingNetworkConfig:
    """Returns the default values for the TemplateMatchingNetworkConfig"""
    return TemplateMatchingNetworkConfig.load(
        os.path.join(
            os.path.dirname(__file__),
            'configs',
            f'{DEFAULT_TMN_CONFIG_FILE}.{CONFIG_EXT}'
        )
    )


def make_default_binary_params(
    cond_id,
    max_template_delay,
    pct_dendrite_spikes,
    template_onset_to_template_prediction_max_delay
):
    """Create a default set of network parameters for the Binary module."""
    ##### Network Parameters #####
    net_params = {}
    # general parameters
    net_params['templates_per_syllable'] = cond_id
    # pre-coincidence layer
    net_params['pre_coincidence_vth'] = int(cond_id * pct_dendrite_spikes / 5)
    net_params['pre_coincidence_dv'] = 100
    net_params['pre_coincidence_refractory_period'] = 10
    # coincidence layer
    net_params['coincidence_dv'] = int(0.3 * 4096)
    net_params['coincidence_vth'] = 11
    # binary prediction layer
    net_params['binary_prediction_dv_soma'] = 1000
    net_params['binary_prediction_dv_dend'] = 100
    # this will need to scale properly w/ weight and num. binary. dets per type
    net_params['binary_prediction_vth_soma'] = 10
    net_params['binary_prediction_vth_dend'] = 8
    # net_params['binary_prediction_soma_refractory_delay'] = 20
    # connections
    net_params['template_onset_to_pre_coincidence_weight'] = 1
    net_params['template_onset_to_pre_coincidence_delay'] = 1
    net_params['pre_coincidence_to_coincidence_weight'] = 10
    net_params['coincidence_to_binary_prediction_weight'] = 48
    net_params['coincidence_to_template_prediction_weight'] = - cond_id
    net_params['binary_feature_to_binary_prediction_weight'] = 11
    net_params['binary_feature_to_binary_prediction_delay'] = max_template_delay
    net_params['template_onset_to_template_prediction_max_delay'] = template_onset_to_template_prediction_max_delay
    net_params['binary_prediction_to_self_weight'] = (
        - net_params['coincidence_to_binary_prediction_weight']
    )
    net_params['binary_prediction_to_decision_weight'] = 11 # decision vth == 10

    return net_params
