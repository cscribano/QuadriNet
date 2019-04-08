# 🐐 Pytorch Base Project
This repo is designed to be a good starting point for every Pytorch project.

## Dummy Dataset
The dummy dataset is composed of pairs (`x`, `y_true`) in which:
* `x`: RGB image of size 128x128 representing a light blue circle (radius = 16 px)
	on a dark background (random circle position, randomly colored dark background)
* `y_true`: copy of `x`, with the light blue circle surrounded with a red line (4 px internale stroke)

| `x` (input)                            | `y_true` (target)                     |
|:--------------------------------------:|:-------------------------------------:|
| ![x0](dataset/samples/sample_x_0.png)  | ![y0](dataset/samples/sample_y_0.png) |
| ![x1](dataset/samples/sample_x_1.png)  | ![y1](dataset/samples/sample_y_1.png) |
| ![x2](dataset/samples/sample_x_2.png)  | ![y2](dataset/samples/sample_y_2.png) |

## Dummy Model
* model input: `x`
* model output: `y_pred`
* loss: MSE between `y_pred` and `y_true`

```
DummyModel(
  (main): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace)
    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Upsample(scale_factor=2, mode=bilinear)
    (9): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (10): ReLU(inplace)
    (11): Upsample(scale_factor=2, mode=bilinear)
    (12): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

## Ouput - Example

```
>> experiment name: magalli
┏━━━━━━━━━━━━━━━━━━━┓
┃ PytorchBase@Capra ┃
┗━━━━━━━━━━━━━━━━━━━┛

▶ Starting Experiment 'magalli' [seed: 4110]
tensorboard --logdir=\\majinbu\Public\capra\log\PytorchBase

[06-28@15:39] Epoch 0.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.015625 │ ↯: 18.73 step/s │ T: 23.41 s
	● AVG Loss on TEST-set: 0.006297 │ T: 8.00 s
[06-28@15:39] Epoch 1.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.004411 │ ↯: 21.05 step/s │ T: 21.64 s
	● AVG Loss on TEST-set: 0.003328 │ T: 8.29 s
[06-28@15:40] Epoch 2.256: │██████████████████████████████████████████████████│ 100.00% │ Loss: 0.002658 │ ↯: 21.24 step/s │ T: 21.70 s
	● AVG Loss on TEST-set: 0.002414 │ T: 8.15 s
[06-28@15:40] Epoch 3.124: │████████████████████████┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈│  48.44% │ Loss: 0.002110 │ ↯: 21.08 step/s
```
