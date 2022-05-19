# DeblurCNN

## Requirements:

- Python 3.10+
- ```pip install -r requirements.txt```


## Run the whole thing

First, you must specify, where your dataset is. Go to ```settings.yaml``` and specify your path.
Example: GOPRO
    GOPRO
    |test
     |blur
      |folders with images
    |train
     |blur
      |folders with images
    Then, your path should be "path_to_GOPRO"/GOPRO/train/blur"

If the dataset path has been specified, run ```py main.py```