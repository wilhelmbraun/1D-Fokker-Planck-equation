# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:02:20 2017

@author: wilhelm
"""


#solve time-dependent FPE for 1D diffusions



############
from dolfin import *
import numpy as np

import matplotlib.pyplot as pl



#############################
#solution of stationary FPE

#different log levels
#CRITICAL  = 50, // errors that may lead to data corruption and suchlike
#ERROR     = 40, // things that go boom
#WARNING   = 30, // things that may go boom later
#INFO      = 20, // information of general interest
#PROGRESS  = 16, // what's happening (broadly)
#TRACE     = 13, // what's happening (in detail)
#DBG       = 10  // sundry

set_log_level(INFO)

ncells_each_direction = 4000

mesh = IntervalMesh(ncells_each_direction, -10.0, 2.0)

h = CellSize(mesh)
plot(mesh)
interactive()






print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
print "::::Computing FPT distribution for 1D diffusion::::"
print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"


sigma = 1.0
tau_a = 5.0
delta_tilde = 10.0
delta =delta_tilde/ tau_a
x_th = 1.0
mu = 0.8


starting_value_adaptation = 0.0
#==============================================================================
# main code
#==============================================================================

#manual mesh refinement in a region of interest    
refining_indicator = 0

if refining_indicator ==1:
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in cells(mesh):
    	p = cell.midpoint()
        
    	if ((p.y()-(starting_value_adaptation))**(2) +((p.x())**(2))  <= 0.01):
            	cell_markers[cell] = True
    
    
    #arbitrary number of refinements
    for k in np.arange(0,1):
    	print "number of refinement iteration", k
    	mesh = refine(mesh, cell_markers)
     
    #look at mesh
    plot(mesh)
    interactive()
    #exit()


V = FunctionSpace(mesh, "CG", 1)


#diffusion matrix

C = sigma/2.



#drift part of the stochastic differential equation system: see notes



#single exponential adaptation current
FX = Expression(('3.0*(mu_in -x[0])'), mu_in = mu, domain = mesh, degree = 1)


###########################################################################################
#implementation of numerical densities
###########################################################################################


#class for nascient delta distribution as starting point for FPT            
class Delta(Expression):
    def __init__(self, eps, degree):
        self.eps = eps
        self.degree = degree
    def eval(self, values, x):
        eps = self.eps
        #values[0] =  ((eps/pi/((x[1]-alpha_adaptation/beta_adaptation)**2 + eps**2))*(eps/pi/((x[0])**2 + eps**2)))/1.4384969697
        #implementation of the nascent delta function: https://en.wikipedia.org/wiki/Dirac_delta_function#Higher_dimensions
        values[0] = (1./(np.sqrt(2.*pi*eps))*exp(-((x[0] - 0.5)**(2))/(2.*eps)))


# Define trial and test function and solution at previous time-step
u = TrialFunction(V)
v = TestFunction(V)
u0 = Function(V)

#print inner(FX, v.dx(0))

u0_function = Delta(1e-4, degree = 1)
u0.interpolate(u0_function)
    
##########################################################################################
#theta scheme, from http://www.karlin.mff.cuni.cz/~blechta/fenics-tutorial/heat/doc.html
##########################################################################################

# Define steady part of the equation
def operator(u, v):
    #return ( K*inner(grad(u), grad(v)) - f*v + dot(b, grad(u))*v )*dx - K*g*v*dsN
    
    #this choice gives us an instability!
    #return (inner(C*grad(u), grad(v)) + v *div(FX*u ) )*dx 
    
    return (dot(C*grad(u), grad(v)))*dx  - u*inner(FX, v.dx(0))*dx



# Time-stepping parameters
T = 5.0
dt = 1e-2
theta = Constant(1.0) # Crank-Nicolson scheme for 0.5
simulation_time = dt


# Test and trial functions
u, v = TrialFunction(V), TestFunction(V)



# Define time discretized equation
F = (1.0/dt)*dot(u-u0, v)*dx + theta*operator(u, v) + (1.0-theta)*operator(u0, v)

#add stabilisation term
#vnorm = sqrt(dot(FX, FX))
#F += (h/(2.0*vnorm))*dot(FX, grad(v))*r*dx


 # Define boundary condition
def boundary(x):
 
    return ((x[0]-x_th) > DOLFIN_EPS )


# Set up boundary condition
bc= DirichletBC(V, Constant(0.0), boundary)



# Prepare solution function and solver
u = Function(V)
problem = LinearVariationalProblem(lhs(F), rhs(F), u, bc)
solver  = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "gmres"
solver.parameters["preconditioner"] = "ilu"
 
u.interpolate(u0)

 
plot(u, title='initial condition')
interactive()
 
# at time t= 0, the CDF is zero!
pde_solution = [0.0, 0.0]
 
progress = Progress("Time-stepping")
set_log_level(PROGRESS)
 
while simulation_time <= T:
 
     if simulation_time % 5 == 0:
         print "solving PDE at time t=", simulation_time
     
     # Solve the problem
     solver.solve()
     
     FPT_probability = 1.- assemble(u*dx)
     
     if simulation_time != 0:
         pde_solution = np.vstack([pde_solution, [simulation_time, FPT_probability]])
         
     #print pde_solution
 
     if simulation_time % 5 == 0:
         print "CDF(t) of FPT =", FPT_probability
     
     #plot solution
     plot(u, title='Solution  at t = %g' % simulation_time)
     #interactive()
 
     # Move to next time step
     u0.assign(u)
     simulation_time += dt
     
     #update progress bar
     print "======================================"
     progress.update(simulation_time / T)
     print "======================================"
     
pl.figure(1)
pl.plot(np.gradient(pde_solution[:,1]))
pl.show()
     
# #save to file using 20 digits after the comma
#==============================================================================
#np.savetxt('./data/FPT_probability_1_theta_23032017_PIF.dat', pde_solution, fmt='%.20f')
