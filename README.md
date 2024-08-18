

# Factor Tensor Network Message Passing (FTNMP)

This repository hosts the Factor Tensor Network Message Passing (FTNMP) package, a specialized tool for calculating the entropy of 3SAT problems using Tensor Network Message Passing (TNMP) on factor graphs.

## Overview

The FTNMP program applies TNMP on a factor graph to solve 3SAT problems, a well-known NP-complete problem in computer science. This approach focuses on calculating the entropy, providing insights into the complexity and characteristics of the 3SAT instances.

## Repository Contents

### FTNMP.ipynb
A comprehensive Jupyter Notebook that introduces the FTNMP framework, detailing the specific mechanisms and processes involved in using TNMP for entropy calculation.

### BP_fast
This module provides vectorized implementations of belief propagation algorithms optimized for solving k-SAT problems.

### TNBP
Contains the essential functions for local contraction and global iteration within the FTNMP process.

### contract_exact
Implements a dynamic slicing method for the exact contraction of tensor networks.

### graph_generate
A utility for generating random k-SAT graphs, as well as graphs that are locally dense but globally sparse.

### get_region_new
Responsible for identifying regions and local subsystems used within the FTNMP algorithm.

### nx_test
Used to detect overlaps in local subsystems and identify intersecting edges.

### Additional Files
These auxiliary files support the main modules by managing various required components.

## FTNMP Folder
This folder demonstrates the capabilities of FTNMP. Key scripts include:

- **main**: Outputs results for R=2, 4, 6, and computes entropy errors using belief propagation, showcasing the algorithm's accuracy and superiority.
  
- **converge_and_steps**: Compares the convergence performance of FTNMP (with R=2 and 3, damping factor = 0.5, and others using default settings) against belief propagation (with damping factor = 0.5 under high local constraint density), highlighting FTNMP's superior convergence in scenarios where graphs are locally dense and globally sparse.
  
- **neighborhood_test**: Analyzes the graph decomposition properties of various types of graphs.

