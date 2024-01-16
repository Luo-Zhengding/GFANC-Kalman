# GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering (GFANC-Kalman)

This is the code of the paper 'GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering' accepted by IEEE Signal Processing Letters. You can find the paper at [Researchgate](https://www.researchgate.net/publication/375695028_GFANC-Kalman_Generative_Fixed-Filter_Active_Noise_Control_with_CNN-Kalman_Filtering) or [Ieee xplore](https://ieeexplore.ieee.org/document/10323505).

<p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-Kalman/assets/95018034/d57c4dfe-84b6-4ec9-bb23-cc4a26531b5e" alt="" width="600" height="500">
</p>
<p align="center">
</p>

<p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-Kalman/assets/95018034/dadf8ba5-9603-4f5c-9dac-24f274de014d" alt="" width="600" height="350">
</p>
<p align="center">
</p>

**HIGHLIGHTS:**

This paper is an improvement upon my ICASSP paper 'Deep Generative Fixed-Filter Active Noise Control (GFANC)'. The code of GFANC: https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control

1. In our ICASSP 2023 paper, GFANC generates its control filter solely using information from the current noise frame, which may lead to inaccuracies when dealing with dynamic noises. Consequently, the focus of this SPL paper is a novel GFANC-Kalman method that improves GFANC by incorporating a computationally efficient CNN-Kalman filtering module.

2. Compared to GFANC in the ICASSP paper, this proposed technique improves the generation of control filters by exploiting the recursive nature of the Kalman filter and the representation-learning capability of CNN, which fully utilizes the correlation between adjacent noise frames.

3. Experiments on real dynamic noises indicate that GFANC-Kalman achieves a higher level of noise cancellation than GFANC, SFANC, and FxLMS. Moreover, GFANC-Kalman exhibits good robustness and transferability when evaluated on real acoustic paths.

**How to use the code:**

You can easily run the "[Main_GFANC_Kalman.ipynb](https://github.com/Luo-Zhengding/GFANC-Kalman/blob/main/Main_GFANC_Kalman.ipynb)" file to get the noise reduction results.

The 1D CNN is trained using a synthetic noise dataset, its label file is 'Hard_Index.csv'. The entire dataset is available at https://drive.google.com/file/d/1hs7_eHITxL16HeugjQoqYFTs-Cm7J-Tq/view?usp=sharing

Especially, the pre-trained sub control filters are obtained on synthetic acoustic paths, where the primary and secondary paths are bandpass filters. If you want to use the GFANC-Kalman system on new acoustic paths only requires obtaining the corresponding broadband control filter and decomposing it into sub control filters. Noticeably, the trained 1D CNN in the GFANC-Kalman system remains unchanged.

**RELATED PAPERS:**
- [Deep Generative Fixed-Filter Active Noise Control](https://arxiv.org/pdf/2303.05788)
- [Delayless Generative Fixed-filter Active Noise Control based on Deep Learning and Bayesian Filter](https://ieeexplore.ieee.org/document/10339836/)
- [GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering](https://ieeexplore.ieee.org/document/10323505)
- [A hybrid sfanc-fxnlms algorithm for active noise control based on deep learning](https://arxiv.org/pdf/2208.08082)
- [Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks](https://arxiv.org/pdf/2208.08440)
- If you are interested in this work, you can read and cite our papers. Thanks!
