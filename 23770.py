# PH30110 Computational Astrophysics - Coursework 1
# Solution based on https://prappleizer.github.io/Tutorials/RK4/RK4_Tutorial.html

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
			# convert to G = 1 units suitable for a solar system
			self.mass = mass.si.value / c.M_sun.value
			self.r_vec = r_vec.si.value / c.au.value
			self.v_vec = v_vec.si.value / 30E3
		else:
			self.mass = mass
			self.r_vec = r_vec
			self.v_vec = v_vec
		if not np.any(self.r_vec):
			self.central_body=True

		self.f_vec = np.concatenate((self.r_vec, self.v_vec))
		self.history = [self.f_vec]

	def return_vec(self):
		'''
		Concatenates the r and v vectors into a single numpy arrays to be used in RK formalism.

		:returns: A numpy array composed of two vectors containing x, y and v_x, v_y components.
		'''
		return self.f_vec

	def set_vec(self, f_new):
		'''
		Sets the phase-space vector.

		:returns: None.
		'''
		self.f_vec = f_new

	def append_history(self, f_vec):
		'''
		Appends the phase-space history vector for a given body.

		:returns: None.
		'''
		self.history.append(f_vec)

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
		for body in bodies:
			if body.central_body:
				central_body = bodies.pop(bodies.index(body))
		try:
			self.central_mass = central_body.return_mass()
		except:
			self.central_mass = None
		self.orbiting_bodies = bodies
		self.nDims = len(self.orbiting_bodies[0].return_vec())
		self.nBodies = len(self.orbiting_bodies)
		self.mass_vec = np.array([body.return_mass() for body in self.orbiting_bodies])
		self.total_mass = self.central_mass + np.sum(self.mass_vec) if self.central_mass is not None else np.sum(self.mass_vec)
		self.name_vec = [i.return_name() for i in self.orbiting_bodies]

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
	
	def set_method(self, rk4_method, **kwargs):
		'''
		Assigns an internal RK4 solver.
		
		:param rk4_method: A function which completes the RK4 routine.

		:returns: None.
		'''
		self.rk4_method = rk4_method

	def rk4(self, t, bodyIndex, dt):
		'''
		Calculates the K values in RK4 integration and returns a new f vector.

		:param t: A time, for differential equations with time-dependence.
		:param bodyIndex: The index of the current orbiting body.
		:param dt: A non-adaptive timestep.

		:returns f_new: An updated f vector after the given timestep.
		'''
		f_vec = self.orbiting_bodies[bodyIndex].return_vec()
		k1 = dt * self.calc_diff_eqns(t, bodyIndex, f_vec, self.central_mass,
																	self.orbiting_bodies, self.nDims, **self.diff_eq_kwargs)
		k2 = dt * self.calc_diff_eqns(t + 0.5*dt, bodyIndex, f_vec + 0.5*k1, self.central_mass,
																	self.orbiting_bodies, self.nDims, **self.diff_eq_kwargs)
		k3 = dt * self.calc_diff_eqns(t + 0.5*dt, bodyIndex, f_vec + 0.5*k2, self.central_mass,
																	self.orbiting_bodies, self.nDims, **self.diff_eq_kwargs)
		k4 = dt * self.calc_diff_eqns(t + dt, bodyIndex, f_vec + k3, self.central_mass,
																	self.orbiting_bodies, self.nDims, **self.diff_eq_kwargs)

		f_new = f_vec + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)

		return f_new

	def rk4_adaptive(self, t, bodyIndex, dt):
		'''
		Calculates the K values in RK4 integration and returns a new f vector.

		:param t: A time, for differential equations with time-dependence.
		:param bodyIndex: The index of the current orbiting body.
		:param dt: The current timestep.

		:returns f_new: An updated f vector after the given timestep.
		'''
		dt_new = self.calculate_step(bodyIndex, dt)
		f_new = self.rk4(0, bodyIndex, dt_new)
		self.dt = dt_new

		return f_new

	def calculate_step(self, bodyIndex, dt):
		'''
		Calculates the step error in RK4 integration and returns a new stepsize.

		:param bodyIndex: The index of the current orbiting body.
		:param dt: The current timestep.

		:returns dt_new: An updated timestep.
		'''
		v1 = np.linalg.norm(np.split(self.rk4(0, bodyIndex, dt), self.nDims/2)[1])
		v2 = np.linalg.norm(np.split(self.rk4(0, bodyIndex, 2*dt), self.nDims/2)[1])
		steperr = np.abs(v1 - v2) / 30
		if steperr > self.relerr:
			dt_new = self.calculate_step(bodyIndex, dt*((self.relerr/steperr)**0.6))
		else:
			dt_new = 2*dt
		return dt_new

	def calculate_mass_centre(self, mass, f):
		'''
		Calculates the centre of mass frame at each timestep.

		:param mass: The mass/masses under observation.
		:param f: The corresponding phase-space vector(s).

		:returns mass_centre: The centre of mass frame data at the given timestep.
		'''
		return mass * f / self.total_mass

	def run(self, T, initstep, relerr=1E-5, adaptive=True):
		'''
		Runs the simulation on a given set of bodies and stores in attribute history.

		:param T: The total time to run the simulation.
		:param initstep: An initial timestep to advance the simulation.
		:param relerr: The relative error tolerance for each timestep.
		:param adaptive: A flag to toggle use of adaptive RK4.

		:returns: None.
		'''
		if not hasattr(self,'calc_diff_eqns'):
			raise AttributeError('You must set a differential equation solver first.')
		if self.has_units:
			initstep = (initstep.si.value * 2*np.pi) / (1*u.yr).si.value
			T = (T.si.value * 2*np.pi) / (1*u.yr).si.value
		if adaptive:
			if self.has_units:
				try:
					_ = relerr.unit
				except:
					relerr = (relerr*u.m/u.s).si.value / 30E3
			self.set_method(self.rk4_adaptive)
			self.relerr = relerr
		else:
			self.set_method(self.rk4)

		# use standard lists to store initial histories because appending to np.array is costly
		self.history, self.mass_centre = [], []
		for body in self.orbiting_bodies:
			self.history.append(np.array(body.return_vec()))
			# self.mass_centre.append(np.array(self.calculate_mass_centre(body.return_mass(), body.return_vec())))

		self.dt = initstep
		clock_time = 0
		start_time = time.time()
		step = 0
		while clock_time < T:
			sys.stdout.flush()
			sys.stdout.write('Integrating: step = {} '.format(step) + '\r')
			for i in range(self.nBodies):
				body = self.orbiting_bodies[i]
				f_new = self.rk4_method(0, i, self.dt)
				body.set_vec(f_new)
				self.history.append(body.return_vec())
				# mass_centre = self.calculate_mass_centre(body.return_mass(), body.return_vec())
				# self.mass_centre.append(mass_centre)
			clock_time += self.dt
			step +=1
		runtime = time.time() - start_time
		print('\n')
		print('Simulation completed in {} seconds'.format(round(runtime,2)))
		self.history = np.array(self.history)
		# self.mass_centre = np.array(self.mass_centre)

	def plot(self):
		'''
		Plots the data stored in attribute history.

		:returns: None.
		'''
		if not hasattr(self,'history'):
			raise AttributeError('You must run a simulation first.')
		data = np.column_stack(self.history)
		# mass_centre_data = np.column_stack(self.mass_centre)
		# position_data, velocity_data = np.split(data - mass_centre_data, 2)
		position_data, velocity_data = np.split(data, 2)
		weights = np.hypot(velocity_data[0], velocity_data[1])
		fig, ax = plt.subplots(figsize=(5, 3))
		cm = plt.cm.get_cmap('jet')
		sc = plt.scatter(position_data[0], position_data[1], c=weights, cmap=cm)
		plt.plot(0, 0, ".k", markersize=12)
		plt.colorbar(sc, label='$v$ [30 km/s]', orientation='vertical')
		nameString = ', '.join(self.name_vec)
		ax.set_title('RK4 Orbit: ' + nameString)
		ax.set_xlabel('$x$ [AU]')
		ax.set_ylabel('$y$ [AU]')
		plt.show()

def two_body_solve(t, bodyIndex, f, central_mass, orbiting_bodies, nDims):
	'''
	External solver function which calculates the accelerations on the orbiting body at each timestep.

	:param t: A dummy time, required by rk4() for differential equations with time-dependence.
	:param bodyIndex: A dummy index, required by rk4().
	:param f: The phase-space vector at a given timestep.
	:param central_mass: The mass of the central body.
	:param orbiting_bodies: A dummy vector of orbiting bodies, required by rk4()
	:param nDims: The number of phase-space dimensions.

	:returns incremented_vector: evaluated differential equation, containing velocities and accelerations.
	'''
	midpoint = int(nDims/2)
	position_vector = f[0:midpoint]
	incremented_vector = np.zeros(f.size)
	incremented_vector[0:midpoint] = f[midpoint:nDims]
	r = np.linalg.norm(position_vector)
	# could try/except ZeroDivisionError here and add softening
	incremented_vector[midpoint:nDims] = (-central_mass / r**3) * position_vector
	return incremented_vector

def nbody_solve(t, bodyIndex, f, central_mass, orbiting_bodies, nDims):
	'''
	External solver function which calculates the accelerations on all orbiting bodies at each timestep.

	:param t: A dummy time, required by rk4() for differential equations with time-dependence.
	:param bodyIndex: The index of the current body.
	:param f: The phase-space vector at a given timestep.
	:param central_mass: The mass of the central body.
	:param orbiting_bodies: A vector of interacting bodies.
	:param nDims: The number of phase-space dimensions.

	:returns incremented_vector: Evaluated differential equation, containing velocities and accelerations.
	'''
	nBodies = len(orbiting_bodies)
	midpoint = int(nDims/2)
	incremented_vector = np.zeros(f.size)
	if central_mass is not None:
		incremented_vector += two_body_solve(0, bodyIndex, f, central_mass, orbiting_bodies, nDims)
	for j in range(nBodies):
		if bodyIndex != j:
			interactingBody = orbiting_bodies[j]
			f_tmp = np.append(f[0:midpoint] - interactingBody.return_vec()[0:midpoint], np.zeros(int(nDims/2)))
			accelerations = two_body_solve(0, bodyIndex, f_tmp, interactingBody.return_mass(), orbiting_bodies, nDims)
			incremented_vector[midpoint:nDims] += accelerations[midpoint:nDims]
	return incremented_vector

def perihelion_velocity(semimajor_axis, eccentricity, central_mass, orbiting_mass):
	'''
	External solver function which calculates the initial perihelion velocity for an orbiting body.

	:param semimajor_axis: The semimajor axis of orbit.
	:param eccentricity: The initial eccentricity of the orbit.
	:param central_mass: The mass of the central body.
	:param orbiting_mass: The mass of the orbiting body.

	:returns initial_velocity: Calculated initial velocity array.
	'''
	eccentricity_factor = np.sqrt((1 + eccentricity) / (1 - eccentricity))
	period_squared = (4 * (np.pi ** 2) * (semimajor_axis ** 3)) / (c.G.si.value * (central_mass+orbiting_mass))
	v_per = (2 * np.pi * semimajor_axis * eccentricity_factor) / np.sqrt(period_squared)
	return np.array([0, v_per])*u.m/u.s

Comet = Body(name='Halley\'s Comet',
			r_vec = np.array([5.2E9,0])*u.km,
			v_vec = np.array([0,880])*u.m/u.s,
			mass = (2.2E14*u.kg).si)

planet1_velocity = perihelion_velocity((2.52*u.au).si.value, 0, c.M_sun.si.value, 1E-3*c.M_sun.si.value)
Planet1 = Body(name='Planet 1',
			r_vec = np.array([2.52,0])*u.au,
			v_vec = planet1_velocity,
			mass = 1E-3*c.M_sun.si)

planet2_velocity = perihelion_velocity((5.24*u.au).si.value, 0, c.M_sun.si.value, 4E-2*c.M_sun.si.value)
Planet2 = Body(name='Planet 2',
			r_vec = np.array([5.24,0])*u.au,
			v_vec = planet2_velocity,
			mass = 4E-2*c.M_sun.si)

jupiter_velocity = perihelion_velocity((5.204*u.au).si.value, 0.049, c.M_sun.si.value, 1.898E27)
Jupiter = Body(name='Jupiter',
			r_vec = np.array([5.204,0])*u.au,
			v_vec = jupiter_velocity,
			mass = 1.898E27*u.kg)

saturn_velocity = perihelion_velocity((9.583*u.au).si.value, 0.057, c.M_sun.si.value, 5.683E26)
Saturn = Body(name='Saturn',
			r_vec = np.array([9.583,0])*u.au,
			v_vec = saturn_velocity,
			mass = 5.683E26*u.kg)

Sun = Body(name='Sun',
			r_vec = np.array([0,0])*u.au,
			v_vec = np.array([0,0])*u.m/u.s,
			mass = c.M_sun.si)

# could use sys.argv here to pass cli arguments
bodies = [Saturn, Jupiter, Sun]
simulation = Simulation(bodies)
simulation.set_diff_eq(nbody_solve)
simulation.run(150*u.yr, 0.05*u.yr, 1, False)
simulation.plot()
