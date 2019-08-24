# %% imports
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.system.samplers import DWaveSampler
from helpers.solvers import default_solver
from helpers.embedding import embeddings
from helpers.convert import to_base_ten
from collections import OrderedDict
from dimod import ExactSolver
import dwavebinarycsp as dbc
# from helpers import draw
import minorminer
import itertools

# %% create and gate
and_gate = dbc.factories.and_gate(["x1", "x2", "x3"])
and_csp = dbc.ConstraintSatisfactionProblem('BINARY')
and_csp.add_constraint(and_gate)

and_csp.check({"x1": 1, "x2": 1, "x3": 1})

# %% check the energy function
configurations = []
for (x1, x2, x3) in list(itertools.product([0, 1], repeat=3)):
    E = 3 * x3 + x1 * x2 - 2 * x1 * x3 - 2 * x2 * x3
    configurations.append((E, x1, x2, x3))
configurations.sort()
print("E, x1, x2, x3")
configurations

# %% create BQM (BinaryQuadraticModel) from CSP (ConstraintSatisfactionProblem)
and_bqm = dbc.stitch(and_csp)
and_bqm.remove_offset()
print(and_bqm.linear)
print(and_bqm.quadratic)

# %% solving on CPU
sampler = ExactSolver()
solution = sampler.sample(and_bqm)
list(solution.data())

# %%
P = 21

bP = "{:06b}".format(P)
print(bP)

csp = dbc.factories.multiplication_circuit(3)
print(next(iter(csp.constraints)))

bqm = dbc.stitch(csp, min_classical_gap=.1)
print("p0: ", bqm.linear['p0'])

# %%
p_vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

fixed_variables = dict(zip(reversed(p_vars), "{:06b}".format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)

print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)

# %%
my_solver, my_token = default_solver()

# %%
sampler = DWaveSampler(solver={'qpu': True})  # Some accounts need to replace this line with the next:
_, target_edgelist, target_adjacency = sampler.structure

# %%
embedding = embeddings[sampler.solver.id]
bqm_embedded = embed_bqm(bqm, embedding, target_adjacency, 3.0)

print("Variable a0 in embedded BQM: ", 'a0' in bqm_embedded)
print("First five nodes in QPU graph: ", sampler.structure.nodelist[:5])

# %%
kwargs = {}
if 'num_reads' in sampler.parameters:
    kwargs['num_reads'] = 50
if 'answer_mode' in sampler.parameters:
    kwargs['answer_mode'] = 'histogram'
response = sampler.sample(bqm_embedded, **kwargs)
print("A solution indexed by qubits: \n", next(response.data(fields=['sample'])))

# %%
response = unembed_sampleset(response, embedding, source_bqm=bqm)
print("\nThe solution in problem variables: \n", next(response.data(fields=['sample'])))

# %%
sample = next(response.samples(n=1))
dict(sample)
a, b = to_base_ten(sample)

print("Given integer P={}, found factors a={} and b={}".format(P, a, b))


# %%
def response_to_dict(response):
    results_dict = OrderedDict()
    for sample, energy in response.data(['sample', 'energy']):
        # Convert A and B from binary to decimal
        a, b = to_base_ten(sample)
        # Aggregate results by unique A and B values (ignoring internal circuit variables)
        if (a, b) not in results_dict:
            results_dict[(a, b)] = energy

    return results_dict


results = response_to_dict(response)
results

# %%

embedding = minorminer.find_embedding(bqm.quadratic, target_edgelist)
if bqm and not embedding:
    raise ValueError("no embedding found")

bqm_embedded = embed_bqm(bqm, embedding, target_adjacency, 3.0)

kwargs['num_reads'] = 1000
response = sampler.sample(bqm_embedded, **kwargs)

response = unembed_sampleset(response, embedding, source_bqm=bqm)

results = response_to_dict(response)
results

# %%
for sample, energy, num_occurences in response.data():
    print(sample.values(), energy, num_occurences)
