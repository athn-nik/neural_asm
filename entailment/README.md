# neural_asm
Code for Neural Activation Semantic Models: Computational lexical semantic models of localized neural activations (Coling 2018)
For every experiment the code is separated in different folders
To run the Baseline experiment for FMRI classification do:
` asd`
where
The stable voxels selection takes a lot to run. Although, when the voxels are selected they can be stored and used for further experiments. The stable voxels will eb included in the repository in a following update.

### entailment experiment :
For the whole dataset:
`python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset three`

### Example with pretrained embeddings on the one_common dataset with the p_avg 250 dim neural activations
`python model/snli.py  --pretrained glove.6B.300d.txt --pretrained_dim 300 --dataset one --na p_avg/250.txt --na_dim 250`

To run the similarity experiments:

To run the taxonomy experiments:

Sense classification code will be updated soon. A draft version of it is now available.
