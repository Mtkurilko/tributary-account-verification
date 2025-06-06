# dataset

dataset generator for testing comparison methods

### how to use

#### generate regular dataset with duplicates

```sh
python generate.py 1000 350 -d 0.3
```

this generates a dataset of 1000 people, ~350 connections, with a potential overlap of 30% of accounts being duplicates.

#### generate identity sequences for ml training/testing

```sh
python generate.py --sequence 100 --steps 5 -o sequences.json
```

this generates 100 identity sequences, each showing 5 evolutionary steps of the same person over time (timestamps increment by 30 days).

### realistic family structures

family relationship generation creates proper genealogical structures:

#### generational modeling
- **age-based generations**: automatically groups family members by birth year (20+ year gaps indicate new generations)
- **realistic parent-child relationships**: enforces 15-50 year age differences
- **sibling relationships**: connects individuals born within 15 years in same generation
- **spouse relationships**: links individuals of similar age across different family lines

#### relationship types
- `parent_child`: directed edges from older to younger generation
- `sibling`: undirected edges within same generation and age range  
- `spouse`: undirected edges between different families, similar ages

### features

- **ground truth tracking**: every profile includes a `base_uuid` field linking to the original identity
- **realistic duplicates**: typos, nicknames, date variations, email changes 
- **proper family trees**: genealogically accurate family relationship structures
- **identity sequences**: track how identities evolve over time with timestamps
- **configurable**: control duplicate likelihood, sequence length, output format
