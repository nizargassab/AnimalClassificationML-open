# AnimalClassificationML

Train an animal image classifier with TensorFlow and Keras. The training command evaluates the best checkpoint on a held-out test split. The prediction command reads the class names saved with that checkpoint.

## Setup

Use Python 3.12 in a virtual environment. Install the dependencies with:

```sh
python -m pip install -r requirements.txt
```

## Train and predict

Place your own licensed images in one folder per class, for example `dataset/cat/` and `dataset/dog/`. Provide at least 10 images per class across at least two classes. The splitter accepts JPEG, PNG and BMP files.

```sh
python dataset_splitter.py dataset
python main.py --epochs 10
python predict.py path/to/animal.jpg
```

The splitter creates `new_dataset/` with a reproducible, approximately 80/10/10 training, validation and test split. If you use `--output` to change the split location, pass the same location to `main.py --data`.

Training creates `model-output/` with `best_model.keras` and `class_names.json`. Use `main.py --output` to save a new run in another directory, and `predict.py --model-dir` to read it. The optional `main.py --transfer-learning` flag downloads MobileNetV2 ImageNet weights during training.

Use separate photos of different animals or scenes for training and testing. Near-duplicates across splits can make the reported accuracy misleading. Model confidence is the maximum output probability for the supplied image.

## License

The source code is available under the [MIT License](LICENSE). Bring your own dataset and check its license before use or redistribution.
