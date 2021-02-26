import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import sys
import time

class Body():
	def __init__(self, mass, r_vec, v_vec, name=None, has_units=True, central_body=False):
		'''
		Initialises an instance of the Body class to be used in orbital simulations.
		
		:param mass: mass of body as an Astropy Quantity if has_units=True, otherwise a float.
		:param r_vec: a vector containing the x, y initial positions of the body.
		:param v_vec: a vector containing the v_x, v_y initial velocities of the body. 
		:param name: a string to store a name to be used in plots.
		:param has_units: determines whether to use Astropy units.

		:returns: None.
		'''
		self.name = name
		self.has_units = has_units
		self.central_body = central_body
		if self.has_units:
			self.mass = mass.cgs.value
			self.r_vec = r_vec.cgs.value
			self.v_vec = v_vec.cgs.value
		else:
			self.mass = mass
			self.r_vec = r_vec
			self.v_vec = v_vec
		if not np.any(self.r_vec):
			self.central_body=True

	def return_vec(self):
		'''
		Concatenates the r and v vectors into a single numpy arrays to be used in RK formalism.

		:returns: A numpy array composed of two vectors containing x, y and v_x, v_y components.
		'''
		return np.concatenate((self.r_vec, self.v_vec))

	def return_mass(self):
		return self.mass

	def return_name(self):
		return self.name

class Simulation():
	def __init__(self, bodies, has_units=True):  
		'''
		Initialises an instance of the Simulation class.

		:param bodies: a list of Body() objects.
		:param has_units: determines whether the bodies use Astropy units.

		:returns: None.
		'''
		self.has_units = has_units
		self.bodies = bodies
		self.nDims = len(self.bodies[0].return_vec())
		self.orbiting_body = self.bodies[0] if not self.bodies[0].central_body else self.bodies[1]
		self.f_vec = self.orbiting_body.return_vec()
		self.name_vec = [i.return_name() for i in self.bodies]

	def set_diff_eq(self, calc_diff_eqns, **kwargs):
		'''
		Assigns an external solver function as the differential equation solver for RK4. 
		For N-body or gravitational setups, this is the function which calculates accelerations.
		
		:param calc_diff_eqns: A function which returns the f vector for RK4.
		:param **kwargs: Additional arguments required by the external function.

		:returns: None.
		'''
		self.diff_eq_kwargs = kwargs
		self.calc_diff_eqns = calc_diff_eqns

	def rk4(self, t, dt):
		'''
		Calculates the K values in RK4 integration and returns a new f vector.

		:param t: A time, for differential equations with time-dependence.
		:param dt: A non-adaptive timestep.

		:returns f_new: An updated f vector after the given timestep.
		'''
		k1 = dt * self.calc_diff_eqns(t, self.f_vec, None, **self.diff_eq_kwargs)
		k2 = dt * self.calc_diff_eqns(t + 0.5*dt, self.f_vec, 0.5*k1, **self.diff_eq_kwargs)
		k3 = dt * self.calc_diff_eqns(t + 0.5*dt, self.f_vec, 0.5*k2, **self.diff_eq_kwargs)
		k4 = dt * self.calc_diff_eqns(t + dt, self.f_vec, k3, **self.diff_eq_kwargs)

		f_new = self.f_vec + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)

		return f_new

	def run(self, T, nSteps, t0=0):
		'''
		Runs the simulation on a given set of bodies and stores in attribute history.

		:param T: The total time to run the simulation.
		:param nSteps: The number of timesteps to advance the simulation.
		:param t0: An optional non-zero start time.

		:returns: None.
		'''
		if not hasattr(self,'calc_diff_eqns'):
			raise AttributeError('You must set a differential equation solver first.')
		if self.has_units:
			try:
				_ = t0.unit
			except:
				t0=(t0*T.unit).cgs.value
			T = T.cgs.value
				
		self.history = [self.f_vec]
		clock_time = t0
		dt = T / nSteps
		print("dt:  {}\n".format(dt))
		start_time = time.time()
		for step in range(nSteps):
			sys.stdout.flush()
			sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(step+1,nSteps,round(clock_time,3)) + '\r')
			f_new = self.rk4(clock_time,dt)
			self.history.append(f_new)
			self.f_vec = f_new
			clock_time += dt
		runtime = time.time() - start_time
		print('\n')
		print('Simulation completed in {} seconds'.format(runtime))
		self.history = np.array(self.history)

	def plot(self):
		'''
		Plots the data stored in attribute history.

		:returns: None.
		'''
		if not hasattr(self,'history'):
			raise AttributeError('You must a simulation first.')
		data = np.column_stack(self.history)
		weights = np.hypot(data[2], data[3])
		fig, ax = plt.subplots(figsize=(5, 3))
		cm = plt.cm.get_cmap('jet')
		sc = plt.scatter(data[0], data[1], c=weights, cmap=cm)
		plt.plot(0, 0, "*k", markersize=12)
		plt.colorbar(sc, orientation='vertical')
		ax.set_title('RK4 Orbit')
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		plt.show()

'''
External solver function which calculates the accelerations on the orbiting body at each timestep.

:param t: A dummy time, required by rk4() for differential equations with time-dependence.
:param f: The f vector at a given timestep.
:param central_mass: The mass of the central body.
:param nDims: The number of phase space dimensions.

:returns incremented_vector: evaluated differential equation, containing velocities and accelerations.
'''
def two_body_solve(t, f, f_increment, central_mass, nDims):
	orbital_position, orbital_velocity = np.split(f, nDims/2)
	if f_increment is not None:
		position_increment, velocity_increment = np.split(f_increment, nDims/2)
		incremented_position = np.add(orbital_position, position_increment)
		incremented_velocity = np.add(orbital_velocity, velocity_increment)
	else:
		incremented_position, incremented_velocity = orbital_position, orbital_velocity
	r = np.linalg.norm(incremented_position)
	orbital_acceleration = (-c.G.cgs.value * central_mass / r**3) * incremented_position
	incremented_vector = np.concatenate((incremented_velocity, orbital_acceleration))
	return incremented_vector

Comet = Body(name='Halley\'s Comet',
			r_vec = np.array([5.2E9,0])*u.km,
			v_vec = np.array([0,880])*u.m/u.s,
			mass = (2.2E14*u.kg).si)

Sun = Body(name='Sun',
			r_vec = np.array([0,0])*u.AU,
			v_vec = np.array([0,0])*u.m/u.s,
			mass = c.M_sun.si)

bodies = [Comet, Sun]
simulation = Simulation(bodies)
simulation.set_diff_eq(two_body_solve, central_mass=Sun.mass, nDims=simulation.nDims)
simulation.run(75*u.yr, 25)
simulation.plot()
