# The Liver Tumor Segmentation
## Introduction
This program use the spike p system to segmentate liver tumor. The data is from the [3Dircadb](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/). But I transform them to .png and add them ct images.
## The usage
Just run python main.py. In addition, you can add argument. These arguments are here.

```python
parse.add_argument("--action", type=str, help="train or test", default="train")
parse.add_argument("--batch_size", type=int, default=2)
parse.add_argument("--learn_rate", type=float, default=1e-4)
parse.add_argument("--num_epochs", type=int, default=20)
parse.add_argument("--ckp", type=str, help="the path of model weight file")
```

## The having trained model

You can click [here](https://drive.google.com/drive/folders/10ou90KTTKUOdPIwR3bh7Bxy8uQkG21dG?usp=share_link) to download the having trained model. And then you need make save_model dir and put it into the dir.

