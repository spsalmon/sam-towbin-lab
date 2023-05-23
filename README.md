# Segment Anything (for towbin lab)

This is a version of Segment Anything where I deleted things I thought to be useless and confusing and created a script allowing members of the lab to easily run the model on their images.

## How to use it ?

First, create a dedicated python virtual environment

```bash
python3 -m venv ~/env_directory/segment_anything/
```

Then activate the environment you just created

```bash
source ~/env_directory/segment_anything/bin/activate
```

Download the code

```bash
git clone git@github.com:spsalmon/sam-towbin-lab.git
```

Install segment-anything

```bash
cd segment-anything-for-towbinlab
pip3 install -e .
```

Install the required libraries

```bash
pip3 install -r requirements.txt
```

Dowload the checkpoint and place it in the models folder, you can find the check point [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

To run the model on your images, change the input and output folders in the script predict_masks.sh, and change the model parameters if necessary. Then just do

```bash
sbatch scripts/predict_masks.sh
```
