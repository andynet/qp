#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:30:30 2019

@author: andy

minimal example inspired by:
https://docs.ocean.dwavesys.com/en/latest/examples/scheduling.html
"""

# %% imports
import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


# %% define ConstraintSatisfactionProblem
def and_gate(x1, x2, y1):
    if x1 and x2:
        return y1
    else:
        return not y1


# %% create csp object and add constraint
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
csp.add_constraint(and_gate, ['x1', 'x2', 'y1'])

# %% convert csp to BinaryQuadraticModel
bqm = dwavebinarycsp.stitch(csp)

# %% create sampler and embeddings
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample(bqm, num_reads=5000)

# %% return result
for sample, energy, occurrences in response.data(['sample', 'energy', 'num_occurrences']):
    print(sample['x1'], sample['x2'], sample['y1'], energy, occurrences)
