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

this generates 100 identity sequences, each showing 5 evolutionary steps of the same person over time (timestamps increment by 30 days). Perfect for testing entity resolution and sequence modeling.

### features

- **ground truth tracking**: every profile includes a `base_uuid` field linking to the original identity
- **realistic duplicates**: typos, nicknames, date variations, email changes 
- **identity sequences**: track how identities evolve over time with timestamps
- **configurable**: control duplicate likelihood, sequence length, output format
