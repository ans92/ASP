# Attention based Simple Primitives

Here is the PyTorch code for our project ASP, Attention based Simple Primitives for Open World Compositional Zero-Shot Learning. The code includes the ASP method implementation on three datasets: UT-Zappos, MIT-States, and CGQA.

![asp diagram_V6](https://github.com/user-attachments/assets/80c11992-c186-4adb-836a-a49b3fc8e7e1)

# Setup

1. Clone the repo
2. We suggest utilizing Anaconda for setting up the environment (It is not mandatory though). In order to establish the environment and activate it, please execute:
```
conda env create --file environment.yml
conda activate czsl
```
3. Navigate to the cloned repository and launch a terminal. Obtain the datasets and embeddings, indicating the preferred location for storage (for example, using DATA_ROOT in this scenario).
```
bash ./utils/download_data.sh DATA_ROOT
mkdir logs
```

# Training

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

# Test

To test a model run the following command:
```
python test.py --logpath LOG_DIR --open_world --fast
```
```LOG_DIR``` is a path where logs are stored during training.
