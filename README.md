## Temporal Graph Convolutional Network (TGCN)

**Ongoing Research Project**

This project aims to develop a model for estimating pain score from videos by utilizing landmarks extracted from each frame as nodes in a static graph. The sequence of graphs in each video forms a temporal sequence. 
For this purpose, we employ TGCN, specifically A3TGCN, which is an attention-based temporal graph convolutional network. More information about TGCN can be found in the PyTorch Geometric Temporal documentation.

# Installation:
1. Clone the repository.

2. Create a new environment and install the required dependencies using  `pip install -r requirements.txt `.
3. Activate the environment, navigate to the tgcn directory, and run  `python main.py `.
4. To run an experiment, check the  `name_exp ` variable in the  `main.py ` file.
Experiment configurations are located in the  `config ` directory. You can create a new YAML file for a new experiment.
# Dataset:
We utilize the Biovid PartA dataset for this research. 

Landmarks are extracted using the dlib library and the Mediapipe framework. For more information, refer to the  `utils.py ` file.

# Dependency:
This project requires installtion of 
1. pytorch
`conda install python=3.7 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

2. pytorch geometric 
`conda install pyg -c pyg`

3. pytorch geometric temporal replace torch-1.13.1+cu117 by the version of pytorch and cuda installed. This command valid for pytorch version 1.13.1, and cuda version 11.7

`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html`


`pip install torch-geometric-temporal`