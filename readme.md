

# Neural Phase Picker models trained on JMA unified dataset

## Overview

We release neural phase picker models—deep learning models designed for seismic waveform arrival-time reading—along with their weight parameters trained using the JMA (Japan Meteorological Agency) Unified Catalogs, which are produced by the Japan Meteorological Agency in collaboration with the Ministry of Education, Culture, Sports, Science and Technology (MEXT). The currently released models and corresponding weights are as follows:

## 01: Naoi et al. (2024, Earth, Planets and Space)

- The related paper is 10.1186/s40623-024-02091-8

- Initial release: 2024-06-28

- Directory: models01/

- This study trained PhaseNet (Zhu and Beroza, 2019) and PhaseNetWC, which doubles the number of channels in each layer of the original PhaseNet architecture, using eight years of JMA Unified Catalog data.

- The released package includes model weights, sample programs, and example waveform data for using the trained models.

- The test program employs the PhaseNet architecture implemented in the SeisBench framework (Woollman et al., 2022; 10.1785/0220210324), and PhaseNetWC is a modification of this architecture. Consequently, the operational environment for these models depends on SeisBench.

- The models and weights have been integrated into the current SeisBench library. Sample programs are provided to demonstrate how to load the weights stored within SeisBench.

- A TensorFlow–Keras implementation of PhaseNet, together with a sample program to use the JMA-trained PhaseNet weights, is also provided (released on 2025-11-01).

- The related paper is 10.1186/s40623-024-02091-8

- Initial release: 2024-06-28

- Directory: models01/

- This study trained PhaseNet (Zhu and Beroza, 2019) and PhaseNetWC, which doubles the number of channels in each layer of the original PhaseNet architecture, using eight years of JMA Unified Catalog data.

- The released package includes model weights, sample programs, and example waveform data for using the trained models.

- The test program employs the PhaseNet architecture implemented in the SeisBench framework (Woollman et al., 2022; 10.1785/0220210324), and PhaseNetWC is a modification of this architecture. Consequently, the operational environment for these models depends on SeisBench.

- The models and weights have been integrated into the current SeisBench library. Sample programs are provided to demonstrate how to load the weights stored within SeisBench.

- A TensorFlow–Keras implementation of PhaseNet, together with a sample program to use the JMA-trained PhaseNet weights, is also provided (released on 2025-11-01).

- The models assume that waveform input is ordered in UNE channel order, and the output is returned in PSN order.

- The training dataset includes a range of event types, including LFEs (low-frequency earthquakes), as described in Naoi et al. (2024).

  

---

## 02: Naoi et al. (under review in Earth, Planets and Space )
- Initial release: 2025-11-21
- Directory: models02/
- Three models (Base model, PhaseNeXt-S, and PhaseNeXt-M suggested in Naoi et al. (in prep.)) trained using 20 years of JMA Unified Catalog data are released. Both model files and trained weights are available.
- Note: In the implementations in the model02 directory, the final softmax layer is not included in the models, following standard PyTorch conventions. In contrast, the models in SeisBench include the softmax layer.
- The models in the models02 directory assume fixed-length waveforms of 4096 samples. Since the provided sample waveforms are 3001 samples long, the released inference results were obtained using zero-padded input waveforms.
- The models assume waveform input ordered in UNE channel order, and outputs are returned in PSN order.
- Only events labeled as ordinary earthquakes in the JMA Unified Catalog were used for training.

## Important files

- run_PhaseNetorg_JMAunified.ipynb
  - Sample program to use models01/model_PhaseNet_JMA.pth, which is the weight file for PhaseNet obtained from the training by Naoi et al. (2024)

- run_PhaseNetWC_JMAunified.ipynb
  - Sample program to use models01/model_PhaseNetWC_JMA.pth, which is the weight file for PhaseNetWC obtained from the training by Naoi et al. (2024)

- run_PhaseNeXt.ipynb
  - Sample program to use the models and weights in the models02 directory, where the PyTorch models and weights were obtained through training with the 20-year JMA Unified dataset.

- sample_wv.mat: Seismic waveform records for testing. Those are recorded by the Seismology Laboratory, Graduate School of Science, Hokkaido University.

### Requirements

```
pip install seisbench
pip install torchinfo
pip install tensorflow (to use Keras model)
```

## References

- Naoi et al. (2024) doi: 10.1186/s40623-024-02091-8
- Zhu and Beroza (2019), doi: 10.1093/gji/ggy423
- Woollman et al. (2022), doi: 10.1785/0220210324
