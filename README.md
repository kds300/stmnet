# STMNet
Spike Template Matching Network implemented using the Lava neuromorphic computing framework.

## Contributors
The original code was developed in a private repository, originally authored by:
- [@kds300](https://www.github.com/kds300)
- [@bala-git9](https://www.github.com/bala-git9)

## Usage
This package provides a framework for building Lava networks that perform template matching on spike sequences.
This can be run as an independent network or as a part of larger network.

Check out the [tutorial](./tutorials/stmnet.ipynb) for a simple example.
Currently, the module only supports fixed-pt CPU simulations of the network.

## Requirements
This package requires [lava](https://github.com/lava-nc/lava).
It also requires `numpy` and `scipy`.
There's no requirements file at the moment, since this module should work once lava is set up.

## Contents

### Network
This package contains the network implementation.
It is split into two classes, `TemplateMatchingNetwork` and `BinaryPredictionNetwork`.

### Configuration
This package contains the configuration information for the network.
The configuration is interfaced through the `TemplateMatchingNetworkConfig`.
The function `default_tmn_config()` can be used to load the default configuration file, found under [stmnet/configs/](stmnet/configs/).

### IO
Contains functions for creating input and output modules for the network.
Supports binary input processes and spike output processes.

### Proc
Contains the lava processes and models for the custom neurons used in the network.
The models are only available for fixed-point CPU implementations.
