# Third-party notices

The map simulator, viewpoint graph construction, and informative graph utilities in
`ac_pbgrl/envs/ariadne` are derived from the public implementations of:

- ARiADNE, upstream revision `8d726ad05e6b0c38672f1679baf7fa2b083a0eb0`.
- Deep Reinforcement Learning-based Large-scale Robot Exploration, upstream
  revision `8aabb0d4cc4f05511183567fd6cef4dd786c5168`.
- The quadtree module credits Daniel Lindsley / toastdriven, following the upstream
  source notice.
- Training maps originate from the DRL Robot Exploration dataset referenced by
  the ARiADNE repository.

The upstream repositories did not include a license file in the source snapshot
available to this project. Redistribution and publication must therefore preserve
the original attribution and be checked with the upstream authors before a public
release. New AC-PBGRL modules are kept clearly separated from the derived simulator.

TARE is not vendored. `scripts/external/install_tare.sh` fetches the pinned
`melodic-noetic` revision `44500592b86138257273e0cab264e6a847ccefc7` and retains
its upstream license and history.
