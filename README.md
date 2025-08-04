# SUPRA: Sensitivity-Guided Universal Pruning Rate Allocation for Efficient LLM Compression

## Setup

Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage

The PruneAndTest.sh contains all the commands to replicate the main results (Table1 and Table 2) in our paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.

```sh
For example, to reproduce the llama-1-7b with 50% unstructured sparsity with wanda, run:
./PruneAndTest.sh llama-1-7b-p50-config.txt

./PruneAndTest.sh llama-1-7b-p50-config.txt

./PruneAndTest.sh llama-1-7b-p50-config.txt

./PruneAndTest.sh llama-1-7b-p50-config.txt
```

## Acknowledgement

This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions

Feel free to discuss papers/code with us through issues/emails!
