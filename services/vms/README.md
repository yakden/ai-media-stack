# Iris — Video Management System

> **This service now lives in its own repository → [github.com/yakden/iris](https://github.com/yakden/iris)**

**Iris** is a self-hosted, single-GPU VMS with cross-camera, **face-anchored** person & object
re-identification. It started here, inside this platform, and was promoted to a standalone,
fully-documented project — README in **English · Русский · Polski**, plus deep dives on the
architecture, the re-identification engine, the REST API, the security model and deployment.

- **Repository:** https://github.com/yakden/iris
- **Highlights:** face-anchored cross-camera re-ID · non-blocking three-thread per-camera pipeline ·
  low-latency live monitoring with two-finger zoom · manual recording · fail-closed security ·
  non-root, capability-dropped container.

It still runs on the same shared NVIDIA T4 alongside the rest of this platform — only the source and
documentation moved to their own home.
