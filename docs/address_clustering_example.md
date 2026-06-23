# Address Clustering Example

Address clustering is a co-location signal. It does not prove ownership, but it
can identify entities that should be reviewed together because they share the
same registered address as an entity with known parent evidence.

## Simple Example

Suppose three Malaysian entities share the same registered address:

```text
Entity A: Nestle Products Sdn. Bhd.
Address: Level 22, Menara Surian, 47810

Entity B: Nestle Manufacturing Malaysia Sdn. Bhd.
Address: Level 22, Menara Surian, 47810

Entity C: Nestle ASEAN Malaysia Sdn. Bhd.
Address: Level 22, Menara Surian, 47810
```

If Entity A has known parent evidence:

```text
Entity A -> Nestle S.A. -> Switzerland
```

but Entity B and Entity C do not report parent data, address clustering says:

```text
B and C share the exact same registered address as A.
They may belong to the same group.
Flag them for review as possible Nestle / Switzerland UIE cases.
```

The pipeline records this as:

```text
uie_source = phase2_address_cluster
evidence_tier = D_address_cluster
```

## Interpretation

Address clustering should be treated as a review cue, not a final UIE
assignment. It means:

```text
These companies sit at the same registered address as a known group entity.
Check whether they share the same parent.
```

It is especially useful for finding possible shared-parent structures among
fund vehicles, special purpose entities, Labuan entities, and companies using
common corporate-secretary addresses.
