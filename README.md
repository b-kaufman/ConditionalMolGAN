# ConditionalMolGAN

A Conditional Generative Advesarial Network for generating molecular graphs with specific properties. Includes supporting code for data processing and setting up hyperparameter sweeps and various evaluation metrics.

certain components of the code are drawn from https://github.com/nicola-decao/MolGAN which supports the publication (https://arxiv.org/abs/1805.11973).

Overview:
To download the dataset used in the original MolGAN paper run data/download_dataset.sh

To convert data to the appropriate format run sparse_molecular_dataset.py. The default arguments are for the above dataset but can be changed if you'd like to try your own sdf or smi files.

to train a GAN use run_GAN.py. There are 3 different types: basic is a reimplementation of the original MolGAN, sim is a self supervised version that uses regularization to try to increase molecular diversity (to see the ways this is done check out similarity.py), and finally cond is the conditional MolGAN.

model parameters are read in from a tsv that must be provided. starting examples for each model are provided in the param_files directory. To generate a param grid for a model you can use params.py, with an example of this for the Conditional MolGAN found at the bottom of the file.

Example Pipeline to try:

cd data
./download_dataset.sh
cd ..
python sparse_molecular_dataset.py
python run_GAN.py cond CGAN_example 0 gdb9_9nodes.sparsedataset    
