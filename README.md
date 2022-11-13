# yesno-example-for-undergraduates

## Model Training
```bash
pip install -r requirements.txt
```

```bash
pip install jupyter
jupyter-nbconvert --to python train.ipynb

cd train
python train.py
```

## Deployment
```bash
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=`realpath ../libtorch` ..
```