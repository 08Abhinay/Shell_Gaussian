"""Stage 1 of the Footwear Fields plan: the controlled test on the CAD set.

The report (``notes/Footwear fields report.pdf``, section 6.1) asks whether
anatomical coordinates make a better category space than object-centric ones.
This package builds what that test needs on top of the finished coordinate,
address and material stages, and runs it. The findings are written up in
``/home/ab5298/Outputs/FootShellGaussian/stage1/REPORT.md``.

Data checks
    winding        generalized winding numbers on the GPU
    mesh_audit     twin faces, open sheets, triangle soups; ``sign_ready``
    signs          winding number vs crossing parity along the fibers
    base_envelope  how much of each shoe the measured shell already explains

The test set
    common         loaders for the flows, canonical anatomy and address books
    geometry       surface sampling and exact point-to-mesh distance
    ring           what a StockX-style 36-view ring sees, and what it shows empty
    dataset        per-shoe samples, targets, A / A+ / B coordinates, regions

The comparison
    train          identical auto-decoders differing only in input coordinates
    run_all        every (variant, split, seed), one queue per GPU
    summarize      result tables for ``train``'s runs
    ringfit        the ring test with carved free-space bounds (the fair one)
    elevation      0 / 15 / 30 degree ablation; footbed vs lining split
    cavity_prior   the anatomy-only cavity: median clearance, no learning
    slices         cross-section figures of held-out shoes

Generated data goes to ``/home/ab5298/Outputs/FootShellGaussian/stage1``; the
pipeline's own outputs are only read.
"""
