# Classification and Uncertainty Quantification of Corrupted Data using Supervised Autoencoders
## Description. 
Parametric and non-parametric classifiers often have to deal with real-world data, where corruptions like noise, occlusions, and blur are unavoidable. We present a probabilistic approach to classify strongly corrupted data and quantify uncertainty, even though the corrupted data does not have to be included to the training data. A supervised autoencoder is the underlying architecture. We use the decoding part as a generative model for realistic data and extend it by convolutions, masking, and additive Gaussian noise to describe imperfections. This constitutes a statistical inference task in terms of the optimal latent space activations of the underlying uncorrupted datum. We solve this problem approximately with Metric Gaussian Variational Inference (MGVI). The supervision of the autoencoder's latent space allows us to classify corrupted data directly under uncertainty with the statistically inferred latent space activations. We show that the derived model uncertainty can be used as a statistical "lie detector" of the classification. Independent of that, the generative model can optimally restore the corrupted datum by decoding the inferred latent space activations.
## Instructions. 
1. Clone Repository

`git clone https://github.com/pjoppich/corrupted_data_classification.git`

2. Install dependencies (assuming pip3 and python3 is installed), prefereably in virtual environment:

`pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_6`

`pip3 install -r ./corrupted_data_classification/requirements.txt`

3. Run code to reproduce main results

`python3 ./corrupted_data_classification/main.py`

4. (Optional) Run code to train and test neural networks (note that this will overwrite the existing neural networks)

- MNIST:

`python3 ./corrupted_data_classification/NNs/MNIST/pretrained_supervised_ae10/autoencoder.py`

- Fashion-MNIST:

`python3 ./corrupted_data_classification/NNs/Fashion-MNIST/pretrained_supervised_ae10/autoencoder.py`


