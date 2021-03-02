### University of Bath PH30110 Computational Astrophysics

This coursework is based on the basic case RK4 two-body problem, with improvements consisting of an adaptive step size and n-body solutions.

Given more time, further improvements could include finishing implementation of the centre of mass frame, softening parameters, and calculation of the total system energy at each timestep (which would provide an indication of the error). One could also employ a numerical integration with time-reversal symmetry, such as the leap-frog or Verlet methods, as this is important in the conext of conservation of energy.

I would also liked to have added unit tests and CI workflow, in addition to passing cli args, but the deadline on this was extremely tight and the question sections were released in parts.