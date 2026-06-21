# Synthetic well logs / ГИС generation — рабочие заметки

## Цель
Сфокусироваться на алгоритмическом ядре образовательного тренажера: генерация управляемых реалистичных LAS/well-log данных с эталонной интерпретацией.

## Найденные классы подходов
1. Physics/rule-based petrophysical forward modeling.
2. Stochastic stratigraphy / Markov facies + petrophysical distributions.
3. Geostatistics / sequential simulation + rock physics.
4. Supervised log synthesis / missing curve prediction.
5. Sequence generative models: TimeGAN/TSGAN, SeqGAN, RNN/TCN.
6. Latent-space generative modeling: autoencoder + MCMC.
7. Diffusion models for multivariate subsurface/elastic parameter synthesis.
8. Foundation/self-supervised models for future QC and evaluator, not MVP generator.

## Ключевой вывод для MVP
Не начинать с GAN/diffusion. Начать с hybrid generator:
- geological scenario schema;
- Markov/grammar-based vertical facies generator;
- per-facies petrophysical property sampler;
- deterministic/empirical forward equations for GR, RHOB, NPHI, DT, RT;
- noise/artifact/degradation module;
- hidden ground truth: tops, facies, Vsh, PHI, Sw, net reservoir/pay;
- export LAS + answer JSON.

## Причина
Образовательному продукту нужна контролируемость, объяснимость и эталонный ответ, а не максимальная статистическая похожесть на реальные данные. Deep generative модели можно добавить позже как calibration/augmentation layer.
