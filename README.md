# Transformers based World Models

Sai Tanmay Reddy Chakkera

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. For a quick summary of the paper and some additional experiments, visit the [github page](https://ctallec.github.io/world-models/).

Implementation adapted from "world-models" 2018. https://github.com/ctallec/world-models

## Prerequisites

The implementation is based on Python3 and PyTorch, check their website [here](https://pytorch.org) for installation instructions. The rest of the requirements is included in the [environment file](environment.yml), to install them:
```bash
conda env create --file environment.yml
```

## Running the experiments

There are 3 experiments in this repository:
* Original world-models implementation
* Transformer DE based implementation
* TraDE based implementation

Each of these is composed of three parts:

  1. A Variational Auto-Encoder (VAE), whose task is to compress the input images into a compact latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN)/Transformer/TraDE, trained to predict the latent encoding of the next frame given past latent encodings and actions.
  3. A linear Controller (C), which takes both the latent encoding of the current frame, and the hidden state of the MDN-RNN given past latents and actions as input and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)) from the `cma` python package.

In the given code, all three sections are trained separately, using the scripts `trainvae.py`, `trainmdrnn.py`, `traintransformersde.py`and `traincontroller.py`.

Training scripts take as argument:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override a model in *logdir* instead of reloading it, add this option.

### 1. Data generation
Before launching the VAE and MDN-RNN/TransformerDE/TraDE training scripts, you need to generate a dataset of random rollouts and place it in the `datasets/carracing` folder. Note that you must copy the generated files from one experiment folder to another. 

Data generation is handled through the `data/generation_script.py` script, e.g.
```bash
cd world-models/
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```
Now copy the generated files to other experiment directories. 
```bash
cp -r datasets/carracing/ ../transformers-wm/datasets/carracing/
cp -r datasets/carracing/ ../trade/datasets/carracing/
```
Rollouts are generated using a *brownian* random policy, instead of the *white noise* random `action_space.sample()` policy from gym, providing more consistent rollouts.

### 2. Training the VAE
The VAE is trained using the `trainvae.py` file, e.g.
```bash
cd world-models/
python trainvae.py --logdir exp_dir
```
VAE only needs to be trained once. So you can copy the files from one experiment directory to another. 
```bash
cp -r exp_dir/vae/ ../transformers-wm/exp_dir/
cp -r exp_dir/vae/ ../trade/exp_dir
```

### 3.1. Training the MDN-RNN
The MDN-RNN is trained using the `trainmdrnn.py` file, e.g.
```bash
cd world-models/
python trainmdrnn.py --logdir exp_dir
```
### 3.1. Training the transformers-wm
The transformers density estimator is trained using the `traintransformersde.py` file, e.g.
```bash
cd transformers-wm/
python traintransformersde.py --logdir exp_dir
```
### 3.2. Training the TraDE
The TraDE model is trained using the `traintransformersde.py` file, e.g.
```bash
cd trade/
python traintransformersde.py --logdir exp_dir
```

A VAE must have been trained in the same `exp_dir` for these scripts to work in step 3. 

### 4. Training and testing the Controller
Finally, the controller is trained using CMA-ES, e.g.
```bash
cd world-models/
python traincontroller.py --logdir exp_dir --n-samples 16 --pop-size 64 --target-return 950 --display
```
Similarly, train the models for each experiment. 

Stop training whenever you see fit. Refer to https://blog.otoro.net/2018/06/09/world-models-experiments/ for approximate train times. 

You can test the obtained policy with `test_controller.py` e.g.
```bash
cd world-models/
python test_controller.py --logdir exp_dir
```
Similarly, test the model for each experiment.

`test_controller.py` also renders the current episode frames in `exp_dir/test_sample/`
Use ffmpeg to convert these frames into a video, e.g.
```bash
cd world-models/exp_dir/test_sample/
ffmpeg -r 30 -f image2 -s 96x96 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test_sample.mp4
```
Video saved as `world-models/exp_dir/test_sample/test_sample.mp4`.

### Notes
When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 16 --pop-size 64 --target-return 950 --display
```
If you do not have a display available and you launch `traincontroller` without
`xvfb-run`, the script will fail silently (but logs are available in
`logdir/tmp`).

Be aware that `traincontroller` requires heavy gpu memory usage when launched
on gpus. To reduce the memory load, you can directly modify the maximum number
of workers by specifying the `--max-workers` argument.

If you have several GPUs available, `traincontroller` will take advantage of
all gpus specified by `CUDA_VISIBLE_DEVICES`.

## Authors

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)

* **Sai Tanmay Reddy Chakkera** [starc52](https://github.com/starc52)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
