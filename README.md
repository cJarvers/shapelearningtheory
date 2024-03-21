# shapelearningtheory
Theoretical investigation on why neural networks fail to learn shape perception

This code implements data generatino, experiments and analyses for the paper
Jarvers & Neumann (in preparation).
Teaching deep networks to see shape: Lessons from a simplified visual world.

## Running the experiments

We use docker to manage dependencies and make the experiment runs reproducible.
To build the docker container, you can run `docker build -t shapelearning .` in 
the project root directory.

Each experiment is provided as a Python script in the `experiments` folder.
For example, to run the first experiment, go to that folder and run

```
python experiment1_bias.py
```

Parameters of the experiments can be controlled with command line arguments.
For example, to run the first experiment on all combinations of shapes,
colors / textures, and image sizes, run:

```
python experiment1_bias.py --shape rectangles --pattern color --imgsize small
python experiment1_bias.py --shape rectangles --pattern striped --imgsize small
python experiment1_bias.py --shape LvT --pattern color --imgsize small
python experiment1_bias.py --shape LvT --pattern striped --imgsize small
python experiment1_bias.py --shape rectangles --pattern color --imgsize large
python experiment1_bias.py --shape rectangles --pattern striped --imgsize large
python experiment1_bias.py --shape LvT --pattern color --imgsize large
python experiment1_bias.py --shape LvT --pattern striped --imgsize large
```

The results of each run are stored in subdirs of `experiments/figures`.