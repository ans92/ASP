# Attention based Simple Primitives

Here is the PyTorch code for our paper ASP, **Attention based Simple Primitives for Open World Compositional Zero-Shot Learning**. The code includes the ASP method implementation on three datasets: UT-Zappos, MIT-States, and CGQA.

### Abstract

<p align="justify">Compositional Zero-Shot Learning (CZSL) aims to predict  unknown compositions made up of attribute and object pairs. Predicting compositions unseen during training is a challenging  task. We are exploring Open World Compositional Zero Shot Learning (OW-CZSL) in this study, where our test space encompasses all potential combinations of attributes and objects. Our approach involves utilizing the self-attention mechanism between attributes and objects to achieve better generalization from seen to unseen compositions. Utilizing a self-attention mechanism facilitates the model's ability to identify relationships between attribute and objects. The similarity between the self-attended textual and visual features is subsequently calculated to generate predictions during the inference phase. The potential test space may encompass implausible object-attribute combinations arising from unrestricted attribute-object pairings. To mitigate this issue, we leverage external knowledge from ConceptNet to restrict the test space to realistic compositions. Our proposed model, Attention-based Simple Primitives (ASP), demonstrates competitive performance, achieving results comparable to the state-of-the-art. </p>

![asp diagram_V6](https://github.com/user-attachments/assets/80c11992-c186-4adb-836a-a49b3fc8e7e1)

## Setup

1. Clone the repo
2. We suggest utilizing Anaconda for setting up the environment (It is not mandatory though). In order to establish the environment and activate it, please execute:
```
conda env create --file environment.yml
conda activate czsl
```
(If you have not created the anaconda environment, then you need to install ```fasttext``` package if you haven't already. Also you need to install the missing libraries while training the model)
```
pip install fasttext
```
3. Navigate to the cloned repository and launch a terminal. Obtain the datasets and embeddings, indicating the preferred location for data storage (for example, using DATA_ROOT in this scenario. If you change the name DATA_ROOT to something else please change that name in flags.py line 3 as well. If you find any error regarding data folder, you may need to update full path of your data folder on line 3 in flags.py file).
```
bash ./utils/download_data.sh DATA_ROOT
mkdir logs
```

## Training

To train the model, following command will be used:
```
python train.py --config CONFIG_FILE
```
where ```CONFIG_FILE``` is the path to the configuration file of the model. The folder ```configs/asp``` contains configuration files for all datasets.

To run ASP on MIT-States, the command is:
```
python train.py --config configs/asp/mit.yml --open_world --fast
```
Replace ```mit.yml``` with ```cgqa.yml``` or ```utzappos.yml``` for other two datasets.

## Test

To test a model run the following command:
```
python test.py --logpath LOG_DIR --open_world --fast
```
```LOG_DIR``` is a path where logs of dataset are stored during training. For example, in case of UT-Zappos dataset your LOG_DIR looks like ```logs/asp/utzappos/```

## Acknowledgement
The project is based on [KG-SP](https://github.com/ExplainableML/KG-SP). Thanks for their awesome works.
