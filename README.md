# Classification and Uncertainty Quantification of Corrupted Data using Semi-Supervised Autoencoders
## Description. 
Often parametric and non-parametric classifiers have to deal with real-world data, where corruptions like noise, occlusions, and blur are unavoidable –- posing significant challenges. We present a probabilistic approach to classify strongly corrupted data and quantify the model and classification's uncertainty, despite the model only having been trained with uncorrupted data. A semi-supervised autoencoder trained on clean data is the underlying architecture. We use the decoding part as a generative model for realistic data and extend it by convolutions, masking, and additive Gaussian noise to describe imperfections. This constitutes a statistical inference task in terms of the optimal latent space activations of the underlying uncorrupted datum. We solve this problem approximately with Metric Gaussian Variational Inference. The supervision of the autoencoder's latent space allows us to classify corrupted data directly under uncertainty with the statistically inferred latent space activations. Furthermore, we demonstrate that the model uncertainty strongly depends on whether the classification is correct or wrong, setting a basis for a statistical "lie detector" of classification. Independent from that, we show that the generative model can optimally restore the uncorrupted datum by decoding the inferred latent space activations.
## Instructions. 
1. Clone Repository

`git clone https://github.com/pjoppich/corrupted_data_classification.git`

2. Install dependencies

`pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_6`

`pip3 install -r ./corrupted_data_classification/requirements.txt --use-deprecated=legacy-resolver`

3. Run code

`python3 ./corrupted_data_classification/main.py`
