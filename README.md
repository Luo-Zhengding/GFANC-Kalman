This is the code of the paper 'GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering' accepted by IEEE Signal Processing Letters.

![Fig1](https://github.com/Luo-Zhengding/GFANC-Kalman/assets/95018034/fed51d95-4cae-4ca3-b5ea-db8c4b054de1)

* You can easily run the "Main_GFANC_Kalman.ipynb" file to get the noise reduction results.
If you are interested in the works, you can read and cite our papers. Thanks!

This paper is an improvement upon my ICASSP paper 'Deep Generative Fixed-Filter Active Noise Control (GFANC)'.
The code of GFANC: https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control

1. In our ICASSP 2023 paper, GFANC generates its control filter solely using information from the current noise frame, which may lead to inaccuracies when dealing with dynamic noises. Consequently, the focus of this SPL paper is a novel GFANC-Kalman method that improves GFANC by incorporating a computationally efficient CNN-Kalman filtering module.

2. Compared to GFANC in the ICASSP paper, this proposed technique improves the generation of control filters by exploiting the recursive nature of the Kalman filter and the representation-learning capability of CNN, which fully utilizes the correlation between adjacent noise frames.

3. Experiments on real dynamic noises indicate that GFANC-Kalman achieves a higher level of noise cancellation than GFANC, SFANC, and FxLMS. Moreover, GFANC-Kalman exhibits good robustness and transferability when evaluated on real acoustic paths.
