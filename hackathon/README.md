# Our approach

## Intro
The idea is that the train-dataset imbalance will strongly guide in the diffusion process the ligands towards the standard pocket and will not predict allo-binding correctly.

To overcome this we employ p2rank - a ML tool that predicts binding pockets. Besides the main one, it can predict other parts of the protein surface that might be suitable for protein-ligand binding.

On the predicted pockets we conidition the boltz-2 diffusion process and then pick the best prediction based on the predicted probability of it being a binder and ipTM.

## Results
We don't know much, but it seems to have helped with predicting the orto-structures even better, but on the allo it still struggled.
