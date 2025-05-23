We provide the following block configurations for reproducing the perplexity results. These configurations can be used with Wanda, BESA, and other related codebases.
Pruning with Wanda:

llama-1-7b-65%:
layer 0 sparsity 0.606471
layer 1 sparsity 0.473868
layer 2 sparsity 0.526883
layer 3 sparsity 0.505138
layer 4 sparsity 0.598658
layer 5 sparsity 0.511989
layer 6 sparsity 0.519802
layer 7 sparsity 0.634315
layer 8 sparsity 0.600351
layer 9 sparsity 0.634275
layer 10 sparsity 0.598428
layer 11 sparsity 0.663331
layer 12 sparsity 0.630639
layer 13 sparsity 0.680940
layer 14 sparsity 0.692909
layer 15 sparsity 0.687250
layer 16 sparsity 0.670893
layer 17 sparsity 0.692869
layer 18 sparsity 0.687540
layer 19 sparsity 0.704127
layer 20 sparsity 0.690465
layer 21 sparsity 0.691426
layer 22 sparsity 0.708033
layer 23 sparsity 0.699239
layer 24 sparsity 0.706551
layer 25 sparsity 0.709976
layer 26 sparsity 0.709956
layer 27 sparsity 0.714383
layer 28 sparsity 0.707322
layer 29 sparsity 0.708994
layer 30 sparsity 0.700701
layer 31 sparsity 0.728776
sparsity sanity check 0.6499
******************************
evaluating on wikitext2
nsamples 166
sample 0
sample 50
sample 100
sample 150
wikitext2 perplexity: 11.174909591674805

llama-1-7b-70%:
layer 0 sparsity 0.642087
layer 1 sparsity 0.514183
layer 2 sparsity 0.612801
layer 3 sparsity 0.503195
layer 4 sparsity 0.556901
layer 5 sparsity 0.676533
layer 6 sparsity 0.561058
layer 7 sparsity 0.721214
layer 8 sparsity 0.640916
layer 9 sparsity 0.637951
layer 10 sparsity 0.706050
layer 11 sparsity 0.721234
layer 12 sparsity 0.721925
layer 13 sparsity 0.737801
layer 14 sparsity 0.756621
layer 15 sparsity 0.748808
layer 16 sparsity 0.727314
layer 17 sparsity 0.732432
layer 18 sparsity 0.734415
layer 19 sparsity 0.759065
layer 20 sparsity 0.739053
layer 21 sparsity 0.747597
layer 22 sparsity 0.748558
layer 23 sparsity 0.745363
layer 24 sparsity 0.750020
layer 25 sparsity 0.752234
layer 26 sparsity 0.750501
layer 27 sparsity 0.754928
layer 28 sparsity 0.746845
layer 29 sparsity 0.750020
layer 30 sparsity 0.740725
layer 31 sparsity 0.757812
sparsity sanity check 0.6999
******************************
evaluating on wikitext2
nsamples 166
sample 0
sample 50
sample 100
sample 150
wikitext2 perplexity: 19.80876922607422


llama-1-7b-75%:
layer 0 sparsity 0.624269
layer 1 sparsity 0.585728
layer 2 sparsity 0.537871
layer 3 sparsity 0.431921
layer 4 sparsity 0.510988
layer 5 sparsity 0.683113
layer 6 sparsity 0.708514
layer 7 sparsity 0.702895
layer 8 sparsity 0.709015
layer 9 sparsity 0.691447
layer 10 sparsity 0.766627
layer 11 sparsity 0.773207
layer 12 sparsity 0.758774
layer 13 sparsity 0.805188
layer 14 sparsity 0.748538
layer 15 sparsity 0.825451
layer 16 sparsity 0.806651
layer 17 sparsity 0.769071
layer 18 sparsity 0.797125
layer 19 sparsity 0.815445
layer 20 sparsity 0.811308
layer 21 sparsity 0.802765
layer 22 sparsity 0.837650
layer 23 sparsity 0.812250
layer 24 sparsity 0.843310
layer 25 sparsity 0.840825
layer 26 sparsity 0.854006
layer 27 sparsity 0.853065
layer 28 sparsity 0.832051
layer 29 sparsity 0.832261
layer 30 sparsity 0.825241
layer 31 sparsity 0.800070
sparsity sanity check 0.7499
******************************
evaluating on wikitext2
nsamples 166
sample 0
sample 50
sample 100
sample 150
wikitext2 perplexity: 56.30166244506836



