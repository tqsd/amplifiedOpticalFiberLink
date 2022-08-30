# amplifiedOpticalFiberLink
This code makes it possible to benchmark the use of Joint Detection Receivers for amplified optical communication links. 
The following options are programmable:
  - link length
  - fiber attenuation
  - photons per pulse (an example shows how to match this to transmission power and baud-rates)
  - number of segments (each segment but the last is assumed to have an amplifier, e.g. an EDFA, at its end)
  - gains
The following can be calculated:
  - spectral efficiency
  - bits per second
  - optimal gains for reaching a given spectral efficiency target
The following output is given:
  - print statements (see examples)
  - visual output (2D plot: spectral efficiency as function of length and number of segments)
  - CSV output (2D plot: spectral efficiency as function of length and number of segments)
  
This code is the basis for numerical investigations in the scientific work
[1] Nötzel, Janis and Rosati, Matteo, "Operating Fiber Networks in the Quantum Limit", https://arxiv.org/abs/2201.12397, doi:10.48550/ARXIV.2201.12397 (2022)
