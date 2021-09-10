# Image Counterfactual Generator

## How to use it
### Using MobileNetV2 Model
To generate a targeted counterfactual for the MobileNetV2 model, you can use the script `gen_t2_mnv2`, which has the mandatory fields described below:
```shell
python gen_t2_mnv2.py --data  ./chihuahua_test/ --output ./cf_region_test  --cclass "French bulldog"
```
Base arguments description:
* --data - The source field where the images are
* --output - The output folder where the CF images will be saved
* --cclass - The name of the counterfactual class being targeted (choose any label from https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt)

Additional arguments:
* --jobs - Number of jobs to run the experiment (DEFAULT = 1)
* --mode - Type of replacement of segmented images, options are: mean, blur, random or inpaint (DEFAULT = blur)
* --timeout - Maximum time allowed to generate a counterfactual (DEFAULT = 60)