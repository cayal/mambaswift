# Mamba.swift
## About
Demonstration of inferencing Albert Gu and Tri Dao's pretrained Mamba models using Apple's MPSGraph.

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
1) Technically speaking, it works

## Performance
* Nonexistent, so be patient and have boatloads of RAM ready while waiting for tokens
* I apologize to the authors, it's entirely unrelated to the Mamba architecture
* This is the worst-performing code I've ever written
* It gives me the impression that I'm doing something wrong
* OTOH Apple Thinks Different™ by providing no real documentation to developers
* I'm also not skilled enough to do a heroic reverse-engineering effort like [this](https://github.com/hollance/neural-engine)
* So I can sleep easy at night in the Code Gulag knowing that nothing could be done to prevent this outcome, and knowing my children and children's children will be mercifully banned from the horrors of programming for 800 years

## TODO
* Improve performance
* Learn to program the GPU directly
* Maybe drop MPSGraph
* Maybe switch careers
