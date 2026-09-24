# anatomical_coordinates

A shared coordinate system for footwear, grounded in the anatomy it wraps.

Shoes have almost nothing in common as shapes — a boot, a sandal and a clog
cannot be registered to one another. But they all surround a foot. So a place
in and around a shoe is described by where it sits relative to the foot, and
the same description then means the same anatomical place on every shoe.

## Running it

```bash
./scripts/run_pipeline.sh                 # 2 GPUs by default
./scripts/run_pipeline.sh --gpus 3
./scripts/run_pipeline.sh --from fit --to join
```

Stages, in order — see `pipeline/stages.py`:

| stage | produces |
|---|---|
| `prepare` | normalized shoe, detected footbed, canonical frame |
| `seat` | an articulated foot resting on that footbed |
| `fit` | foot shape, pose and placement inside the cavity |
| `anatomy` | canonical anatomical regions on the fitted foot |
| `leg` | the lower leg, leaving through the collar |
| `join` | foot and leg as one surface |
| `coordinates` | the shared volumetric map for each shoe |
| `address` | an anatomical address for every footwear point |
| `material` | coverage and material extent at every place on the foot |

## Layout

```
foot_fitter.py     fit a parametric foot into a shoe cavity
cavity_field.py    the shoe's cavity as a differentiable field
losses.py          containment, support, seating, shape priors
config.py          weights and the staged optimizer schedule
supr_torch.py      differentiable body model, gradients intact
shoe_data.py       load one prepared shoe
evaluate.py        the independent exact judge; never used to optimize

coordinate_mapping/   the shared volumetric map (see its own docstring)
pipeline/             the stages and the one runner
tests/                synthetic geometry with analytic answers
docs/                 start at state_of_play.md
```

## Asking it something

Both directions of an address, for any point and any shoe:

```python
from anatomical_coordinates.coordinate_mapping import AddressBook

address = book.query(points)                  # points  -> (u, v, r)
points  = book.place(face, barycentric, r)    # (u, v, r) -> points
```

`(u, v)` is a triangle of the foot surface and a position inside it; `r` is
outward progress along the fiber through that point. The same address means the
same anatomical place in every shoe — sent through all 27 it returns to within
0.006 mm, while the physical points it names sit 35 mm apart.

## Two things worth knowing

**The map is a flow, not a cage.** Each shoe's anatomy is reached from the
canonical anatomy by following a velocity field. Such a map is invertible for
any amount of deformation, so nothing can fold and no limit is needed on how
different a foot may be from the canonical one. The earlier approach stretched
a fixed tetrahedral cage and failed when a cell was about to invert, which
forced feet to be flattened toward the reference until the cage could cope.

**An address means the same place in every shoe.** That claim is checked, not
assumed: `pipeline/address_check.py` sends one set of addresses through all 27
shoes and reads them back.

**Fitting is judged by code that does no fitting.** Every result is scored with
the exact evaluator in `foot_prior`, never with the differentiable loss the
optimizer descends. A loss going down is not evidence.

## Older naming

The research reports number these stages. `pipeline/stages.py` holds the one
mapping table; nothing else in the code refers to the numbers.
