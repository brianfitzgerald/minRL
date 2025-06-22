"""
This is a series of writing experiments.
Take an input dataset of prompts, generate responses, determine the difference between responses, using
different metrics.

The default objective is to take a paragraph, and generate the next paragraph. Then reward based on
the similarity to the original paragraph.

If a dataset isn't found, it's downloaded from Modal.
"""
