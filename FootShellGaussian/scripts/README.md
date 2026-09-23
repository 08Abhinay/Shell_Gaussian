# Pipeline

One command builds shared anatomical coordinates for every shoe in the dataset.

```bash
./scripts/run_pipeline.sh                 # 2 GPUs by default
./scripts/run_pipeline.sh --gpus 3
./scripts/run_pipeline.sh --from fit --to join
./scripts/run_pipeline.sh --shoes crocs leather_boots
```

## What it does

| stage | what it produces | output directory |
|---|---|---|
| `prepare` | normalized shoe, detected footbed, canonical frame | `shoes/` |
| `seat` | an articulated foot resting on that footbed | `seating/` |
| `fit` | foot shape, pose and placement inside the cavity | `foot/` |
| `anatomy` | canonical anatomical regions on the fitted foot | `anatomy/` |
| `leg` | the lower leg, leaving through the collar | `leg/` |
| `join` | foot and leg as one surface | `joined/` |
| `coordinates` | the shared volumetric coordinate map per shoe | `coordinates/` |
| `address` | an anatomical address for every footwear point | `addresses/` |

`canonical/` holds the reference volume, the semantic field and the
correspondence table, built once and shared by every shoe. `status.json`
records which shoes survived each stage and why any failed.

## Addresses

An address says **where on the foot** a point is and **how far out** from it:

| part | meaning |
|---|---|
| `face`, `barycentric` | a triangle of the foot surface and a position inside it - the `(u, v)` |
| `r` | outward progress along the fiber, 0 on the skin, 1 at the outer envelope |
| `arclength` | distance out along that fiber |
| `label` | which region the fiber lands on: foot skin, ankle, lower leg |
| `residual_mm` | how far the address lands from its own point when sent back |

Both directions are available, for any point and any shoe:

```python
from anatomical_coordinates.coordinate_mapping import AddressBook

address = book.query(points)                  # points  -> (u, v, r)
points  = book.place(face, barycentric, r)    # (u, v, r) -> points
```

`addresses/<shoe>/addresses.npz` holds one address per sampled point.
`addresses/cross_shoe.json` records what happens when one address is sent
through all 27 shoes: the physical points differ, the address does not.

## Where the work happens

`prepare`, `seat`, `anatomy` and `join` run the original implementations in
`scripts_legacy/`, unchanged. `fit`, `leg`, `coordinates` and `address` are the
GPU implementations in `anatomical_coordinates/`.

`coordinates` replaces the older tetrahedral cage. The map is the flow of a
velocity field, so it is invertible for any amount of deformation and needs no
constraint on how different a foot may be from the canonical one.

`address` follows fibers, which is the classical way to put coordinates on a
shell: the correspondence map is constant along fibers, so it satisfies a
transport equation with the surface as its boundary condition (Yezzi and
Prince, *An Eulerian PDE approach for computing tissue thickness*, IEEE TMI
2003). Fibers are traced once from every canonical vertex and stored as a
table; a query either reads that table or traces its own fiber, and the two are
compared so the cheap answer is never trusted on argument alone.

## Older naming

The research reports number these stages. The mapping is in
`anatomical_coordinates/pipeline/stages.py`; nothing in the code uses the
numbers.
