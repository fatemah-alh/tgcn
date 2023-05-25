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