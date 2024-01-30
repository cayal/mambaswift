# Mamba.swift
## About
Demonstration of inferencing Albert Gu and Tri Dao's pretrained Mamba models using Apple's MPSGraph.

Based on [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal), 

Usage: 
```
$ cd Mamba/Sources/Demo/Extra
$ poetry install --no-root && poetry shell
$ python -m convert mamba-130m
$ exit
$ cd ../../..
$ swift run Demo "Sources/Demo/Extra/mamba-130m/" -p "My name is" -t 3
~ A FEW HOURS LATER ~
$ ["My", " name", " is", " Tina"]
...
$ ["My", " name", " is", " Tina", " Lloyd"]
```

## Features
1) MPSGraph based inference of `state-spaces/mamba[xxx]` series of models
2) Swift based tokenization

## Caveats
* Using MPSGraph breaks down on the SSM scan step due to MPSGraph's lack of control over buffer allocation
* As a result, the CPU burns frankly incredible amounts of RAM and flops before being able to dispatch to the GPU
* I'm not sure if there's a way to fix performance using MPSGraph

## `chaos`
* Branch for rewriting the entire inference pipeline in a series of Metal compute kernels
* New approach will use heap allocation and a separate scratch heap for each inference context

## TODO
* Make chaos branch build: Swift Package will need a prebuild script for shared C/Swift header and the metal source
* Investigate data races in the earlier-coded pipeline steps now that I'm somewhat familiar with Metal
* Integrate chaos runner after data races definitely gone
