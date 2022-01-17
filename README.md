## Introduction
- This is the training and evaluation code for our work "Lifelong Generative Learning via Knowledge Reconstruction".
- If our paper is accepted, we will publish the full code.

## Requirement
- python = 3.8.5
- torch = 1.7.1
- torchvision = 0.8.2
- lpips
- ignite  

## How to run
### You can test our method by executing the script we provide, or by running the following command.
```sh
# Finetuning on svhn
python main.py -dataset svhn -epoch 10 -method LGLvKR_fine -gpu 0 -generate
```

```sh
# Joint training on Fashion MNIST
python main.py -dataset fashion -epoch 10 -method LGLvKR_joint  -gpu 0 -generate

# Proposed method on MNIST
python main.py -dataset mnist -epoch 10 -method LGLvKR  -gpu 0 -generate

# Proposed method without knowledge reconstruction
python main.py -dataset svhn -epoch 10 -method LGLvKR_noKR  -gpu 0 -generate
```
### arguments
- -method: The name of the method you want to test, you can choose from {`LGLvKR`, `LGLvKR_fine`,  `LGLvKR_joint`, `LGLvKR_noFC`, `LGLvKR_noKR`} currently.
- -generate: Whether images need to be generated, default is `False`.
- -fid:  Whether fid needs to be calculated, default is `False`.
- -ACC:  Whether ACC needs to be calculated, default is `False`.

 



