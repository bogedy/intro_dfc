# intro_dfc
Introspective Deep Feature Consistent Variational Autoencoder

My attempt to implement a [Deep Feature Consistent Variational Autoencoder](https://arxiv.org/abs/1610.00291) but in the introspective style of [this paper](https://arxiv.org/abs/1807.06358). I call it, *The Introspective Deep Feature Consistent Variational Autoencoder*, or if you like word salads, *Autoencoding Variational Bayes Using Self-Supervised High Level Latent Features*, or if you don't, *Introspective DFC VAE*.

This project makes use of the new TensorFlow 2.0 beta using a custom training loop. Man oh man things are easier now!

Model defined in `model.py`, data input done in `data.py`, training functions defined in `train_ops.py`, and the jupyter notebook is for testing things locally with a scaled down model.

todo:
* ~~Clean the code~~ more or less done
* ~~learn how to take advantage of multiple GPUs~~ eh, that was overrated anyways
* ~~Train my normal VAE~~ Done!
* ~~Actually implement it~~ Done! It doesn't work very well yet. But it works.

resources used:
* https://www.tensorflow.org/beta/tutorials/generative/cvae
* https://www.tensorflow.org/tutorials/load_data/images
* https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/variational_autoencoder.ipynb
* In depth step by step on the math of VAEs: https://arxiv.org/abs/1606.05908
