Welcome to ThermoNTFA's documentation!
======================================
.. role:: python(code)
   :language: python
   :class: highlight

-------------------------------------------------------
Thermo-Plastic Nonuniform Transformation Field Analysis
-------------------------------------------------------

In engineering applications, surface modifications of materials can greatly influence the lifetime of parts and structures.
For instance, laser melt injection (LMI) of ceramic particles into a metallic substrate can greatly improve abrasive resistance.
The LMI process is challenging to model due to the rapid temperature changes, which induce high mechanical stresses.
Ultimately, this leads to plastification and residual eigenstresses in particles and matrix. These depend on the process parameters.
In order to predict these stresses, we propose a major extension of the Nonuniform Transformation Field Analysis
that enables the method to cope with strongly varying thermo-elastic material parameters over a large temperature range (here: 300 to 1300K).
The newly proposed $\theta$-NTFA method combines the NTFA with a Galerkin projection to solve for the self-equilibrated fields
needed to gain the NTFA system matrices. For that, we exploit our recent thermo-elastic reduced order model [1]
and extend it to allow for arbitrary polarization strains.
An efficient implementation anda rigorous separation of the derivation of the reduced order model is proposed.
The new $\theta$-NTFA is then validated for various thermo-mechanical loadings and in thermo-mechanical two-scale simulations.

[1] S. Sharba, J. Herb, F. Fritzen, Reduced order homogenization of thermoelastic materials with strong temperature
dependence and comparison to a machine-learned model, Archive of Applied Mechanics 93 (7) (2023) 2855–2876.
doi: `10.1007/s00419-023-02411-6<https://doi.org/10.1007/s00419-023-02411-6>`_

Features
^^^^^^^^

- The **ThermoNTFA** acts as a reduced order model (ROM) and approximates the effective behavior of composite materials that consist of thermoelastic and thermoplastic constituents.
- The material parameters of all constituents are allowed to depend strongly on the temperature.
- This temperature-dependence is reflected in the ROM that is based on interpolated space-saving tabular data at arbitrarily many temperature points.
- Possible application: Eigenstress Analysis of Laser Dispersed Materials

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   workflow
   installation
   api
   examples
   MarkdownLinks/license
   MarkdownLinks/citation
   MarkdownLinks/changelog

.. include:: copyright.rst
