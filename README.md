# Volumetric Fusion

<img src="kitchen.png" width=350px align="right"/>

An implementation of Volumetric TSDF fusion in pure numpy for pedagogical purposes.

- [x] Implement the [Curless and Levoy](https://graphics.stanford.edu/papers/volrange/volrange.pdf) algorithm in pure numpy.
- [ ] Implement the [Newcombe et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf) algorithm in pure numpy.
- [ ] Implement [marching cubes](https://cg.informatik.uni-freiburg.de/intern/seminar/surfaceReconstruction_survey%20of%20marching%20cubes.pdf) from scratch.
- [ ] Implement TSDF volume using a [hash table](https://graphics.stanford.edu/~niessner/papers/2013/4hashing/niessner2013hashing.pdf).

## Installation

1. Create conda env and install dependencies.

```bash
conda create -n fusion python=3.8
conda activate fusion
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run

```bash
python main.py --path <path/to/your/dataset/>
```