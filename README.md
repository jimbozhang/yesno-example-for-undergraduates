# yesno-example-for-undergraduates

## Model Training
```bash
$ pip install -r requirements.txt
```

```bash
$ cd train
$ jupyter-nbconvert --to python train.ipynb
$ python train.py
```

## Deployment
```bash
$ cd deploy
$ ./install_libtorch.sh
```

```bash
$ mkdir build
$ cd build

$ cmake -DCMAKE_PREFIX_PATH=`realpath ../libtorch` ..
$ make
```

```bash
$ ./example-app
usage: example-app <model-path> <wav-path>

$ ./example-app
usage: example-app <model-path> <wav-path>

$ ./example-app ../../train/model.pt ../../train/waves_yesno/0_0_0_0_1_1_1_1.wav
NO NO NO NO YES YES YES YES

$ ./example-app ../../train/model.pt ../../train/waves_yesno/0_0_1_0_1_0_1_1.wav
NO NO YES NO YES NO YES YES

$ ./example-app ../../train/model.pt ../../train/waves_yesno/1_1_1_1_1_1_1_1.wav
YES YES YES YES YES YES YES YES
```