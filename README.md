# Nervus
This is an AI model used for single/multi-label and/or single/multi-class tasks with image and/or tabular data.
Although this has a possibility to apply wide range of fields, we intended to use this model for medical imaging classification task.

Additionally, we merged DeepSurv model [citation] into this model, which is a model that merges Cox proportional hazard model with deep learning. It is a useful model for prognosis estimation by dealing with binary variables such as decease or not, and the period until the event. The original DeepSurv model could only handle tabular data, but we have added images to it.  

Nervus can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.

# Brief Usage
- Directory tree  
Set directories as follows.  

Nervus (this repository)  
   └materials   
　     └imgs (this repository has image files for CNN.)  
 　    └docs  
　　　　  └trials.csv (any name is available if you change `parameters.csv`)

# Detailed Preparation
## CSV
This is the csv which we show as trials.csv in the brief usage section.  
CSV must contain columns named `uniqID`, `label_XXX`, and `split`. Additionally, if you use images as an input, you need `imgpath`.

Example of csv in the docs:
| uniqID |             imgpath            | label_cancer | split |
| -----  | ------------------------------ |  ---------   | ----- |
| 0001   | materials/imgs/png_128/AAA.png | malignant    | train |
| 0002   | materials/imgs/png_128/BBB.png | benign       | val   |
| 0003   | materials/imgs/png_128/CCC.png | malignant    | train |
| 0004   | materials/imgs/png_128/DDD.png | malignant    | test  |
| 0005   | materials/imgs/png_128/EEE.png | benign       | train |
| 0006   | materials/imgs/png_128/FFF.png | malignant    | train |
| 0007   | materials/imgs/png_128/GGG.png | benign       | train |
| 0008   | materials/imgs/png_128/HHH.png | benign       | val   |
| 0009   | materials/imgs/png_128/III.png | malignant    | test  |
| :      | :                              | :            | :     |

Note:
- `uniqID` must be unique.
- `Institution` should be institution name.
- `ExamID` should be unique in each institution.
- `filepath` should have a path to images for the model.
- `label_XXX` should have a classification target. Any name is available. If you use more than two `label_XXX`, it will be automatically recognize multi-label classification and automatically prepare a proper number of classifiers (FCs). 
- `split` should have `train`, `val`, and `test`.
- When you use inputs other than image, `input_XXX` is needed. 
- When you use deepsurv, `periords_XXX` is needed as well.


## Model implemantation
For training and internal validation(tuning), 

`python train.py --`

`python test.py --`

## For many trials
If you need many trials, use `work_all.sh`. In this case, `parameter.csv` must be prepared.

GPU and path to `parameter.csv` should be defined in the `work_all.sh`.
Other parameters are defined in the `parameter.csv`. 

### parameter.csv items
- task: task name
  - example: classification, regression, deepsurv
- csv_name: csv file name contains labeled training data, validation data, and test data
  - default path: ../materials/splits/ (see: align_env.py)
- image_dir: image dataset directory name
  - default path: ../materials/images/ (see: align_env.py)
- model: model name
  - example
    - MLP only: MLP
    - CNN only: B0, B2, B4, B6, ResNet, ResNet18, DenseNet, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase,ConvNeXtLarge, ViTb16_<image_size>, ViTb32_<image_size>, ViTl16_<image_size>, and ViTl32_<image_size>, where <image_size> is the size of input image.
    - MLP+CNN : MLP+B0, MLP+ResNet, MLP+ConvNeXtTiny, MLP+ViTb16_256 ... (combine above)
- criterion: Loss function
  - example: 
    - classification: CEL ※CEL=CrossEntropyLoss
    - regression: MSE, RMSE, MAE
    - deepsurv: NLL
- optimizer: optimization algorithm
  - example: SGD, Adadelta, Adam, RMSprop
- epochs: number of training with entire dataset 
- bach_size: number of training data in each batch
- sampler: samples elements randomly or not.
  - example: yes, no
- augmentation: increase the amount of data by slightly modified copies or created synthetic.
  - example: yes, no
- input_channel: specify the channel of when image is handled, or any of 1 channel(grayscale) and 3 channel(RGB).
  - example:
    - 1 channel(grayscale): 1
    - 3 channel(RGB): 3


# Debugging
## MakeFile
Edit Makefile according to task.

## env
`~/lib/align_env.py`

## Logger
NervusLogger
only to configure logger, usage is the same with default logging module. see [here](https://docs.python.org/3/howto/logging.html).  
The default level is INFO, and handler is StreamHandler.
```py
logger = NervusLogger.get_logger('logger_name')
...
logger.info('info message')
logger.error('error message')
```
when debug. call set_level method.
```py
logger = NervusLogger.get_logger('logger_name')
NervusLogger.set_level(logging.DEBUG)
...
logger.debug('debug message')
```
# CUDA VERSION
CUDA Version = 11.3, 11.4
