# neural_asm
Code for Neural Activation Semantic Models: Computational lexical semantic models of localized neural activations (Coling 2018)
For every experiment the code is separated in different folders
To run the Baseline experiment for FMRI classification do:
` python neural_predictor/neural_bsl.py -p9 -tr 1.0 -st_vox `
where :
* `-p9` refers to the participant (`all` stands for all participants)
* `-tr` whether to train or not (`-notr`)
* `1.0` refers in the fraction of training data to use in order to train/test the code for toy data.
* `-st_vox` whether stable voxel selection should be performed or if they should be loaded.

The stable voxels selection takes a lot to run. Although, when the voxels are selected they can be stored and used for further experiments. The stable voxels will eb included in the repository in a following update.

### source_dsms:
Inside that folder is the code for coocurrence counts extracted from large corpus in order to use as features for neural decoding.
Detailed README and how to is inside that folder.

### entailment experiment :
For the whole dataset:
`python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset three`

##### Example with pretrained embeddings on the one_common dataset with the p_avg 250 dim neural activations
`python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset one --na p_avg/250.txt --na_dim 250`

### To run the similarity experiments:
`python close_loop.py`

### To run the taxonomy experiments:
`python cluster.py dataset_name`

The similarity experiments are not fully automated but if you see the code some paths are hard-coded. It will soon be updated.

Sense classification code will be updated soon. A draft version of it is now available.
