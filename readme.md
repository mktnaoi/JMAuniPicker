

# Neural Phase Picker models trained on JMA unified dataset

## Overview

We release the weight parameters for PhaseNet, a deep-learning model designed for seismic waveform arrival-time reading. The released parameters were obtained by the training using the dataset related to the JMA unified catalog produced by the Japan Meteorological Agency in collaboration with the Ministry of Education, Culture, Sports, Science and Technology (MEXT). Additionally, we are providing the weight parameters for the PhaseNetWC model, which features doubled channels in each layer of the original PhaseNet architecture, along with a test program for the models with the released weight parameters for some test datasets. This test program employs the PhaseNet architecture incorporated into the SeisBench software (Woollman et al. 2022; 10.1785/0220210324), and PhaseNetWC is a modification of this architecture. Consequently, the operational environment for these programs is reliant on Seisbench. 

[241211] These models are reported in Naoi et al. (2024; EPS), and the corresponding weights have been incorporated into the latest Seisbench and are now available!

- model_PhaseNet_JMA.pth: Weight parameter of PhaseNet
- model_PhaseNetWC_JMA.pth: Weight parameter of PhaseNetWC
- run_PhaseNetorg_JMAunified.ipynb: test program to use model_PhaseNet_JMA.pth
- run_PhaseNetWC_JMAunified.ipynb: test program to use model_PhaseNetWC_JMA.pth
- sample_wv.mat: Seismic waveform records for testing. Those are recorded by the Seismology Laboratory, Graduate School of Science, Hokkaido University.

### Requirement

```
pip install seisbench
pip install torchinfo
```

## References

- The paper detailing these released models is published from https://doi.org/10.1186/s40623-024-02091-8.
- Zhu and Beroza (2019), doi: 10.1093/gji/ggy423
- Woollman et al. (2022), doi: 10.1785/ 0220210324