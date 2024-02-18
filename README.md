# KFC: Knowledge Reconstruction and Feedback Consolidation Enable Efficient and Effective Continual Generative Learning

## Introduction

Official training and evaluation codes for the work of "KFC: Knowledge Reconstruction and Feedback Consolidation Enable Efficient and Effective Continual Generative Learning".

## Requirement

- `python = 3.8.5`
- `torch = 1.7.1`
- `torchvision = 0.8.2`
- `lpips`
- `ignite  `

## How to run

Please confirm the dataset is well downloaded or softly linked in the file  "./dataset" at first.

You can test our method by executing the script we provide, or by running the following command in the path `./scripts`

```sh
# on MNIST
bash -i run.sh mnist 0 50
# on FASHION
bash -i run.sh fashion 0 50
# on SVHN
bash -i run.sh svhn 0 50
# on CIFAR
bash -i run.sh cifar 0 50
```

### arguments

- `-method`: The name of the method you want to test, you can choose from {`KFC`, `KFC_fine`,  `KFC_joint`} currently.
- `-generate`: Whether images need to be generated, default is `False`.
- `-fid`:  Whether fid needs to be calculated, default is `False`.
- `-ACC`:  Whether ACC needs to be calculated, default is `False`.

For more tunable arguments, please take a look at the the `main*.py` file.

## License

**Apache License 2.0**

A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

| Permissions         | Conditions                      | Limitations       |
| ------------------- | ------------------------------- | ----------------- |
| ✔️ Commercial use | ⓘ License and copyright notice | ❌  Trademark use |
| ✔️ Modification   | ⓘ State changes                | ❌ Liability      |
| ✔️ Distribution   |                                 | ❌  Warranty      |
| ✔️ Patent use     |                                 |                   |
| ✔️ Private use    |                                 |                   |
