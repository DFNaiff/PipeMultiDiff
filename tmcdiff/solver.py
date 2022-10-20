# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

import numpy as np
import scipy.integrate
import pyequion2

from . import builder


class TransportSolver(object):
    def __init__(self, elements : List[str],
                 activity_model : str = "DEBYE"):
        self.eqsys = pyequion2.EquilibriumBackend(elements,
                                                  from_elements=True,
                                                  backend="torch",
                                                  logbase="e",
                                                  activity_model="DEBYE")
        self.kreaction = None
        
    def set_flow_conditions(self, TK : float,
                            flow_velocity : float,
                            pipe_diameter : float):
        self.TK = TK
        self.flow_velocity = flow_velocity
        self.pipe_diameter = pipe_diameter
        self.shear_velocity = get_shear_velocity(flow_velocity, pipe_diameter, TK)
        self.water_density = pyequion2.water_properties.water_density(TK)
        
    def set_initial_conditions(self, molal_balance : Dict[str, float],
                               solid_phases : List[str]):
        self.initial_molal_balance = molal_balance
        self.solid_phases = solid_phases
    
    def build_transport(self, ngrid, ypmax):
        self.builder = builder.TransportBuilder(
                           self.eqsys,
                           self.TK,
                           self.shear_velocity,
                           self.initial_molal_balance,
                           self.solid_phases,
                           kreaction=self.kreaction)
        self.ngrid = ngrid
        self.ypmax = ypmax
        self.builder.make_grid(ngrid, ypmax)
        self.builder.set_species()
    
    def solve(self, xmax : float, print_frequency=None):
        tmax = xmax/self.flow_velocity
        tarr = []
        logc = []
        xarr = []
        fluxarr = []
        y = np.array([self.initial_molal_balance[el] for el
                      in self.builder.eqsys.solute_elements])
        solver = scipy.integrate.ode(self.f)
        solver.set_integrator("vode")
        solver.set_initial_value(y)
        xarr = [y.copy()]
        tarr = [0.0]
        logc.append(self.builder.get_logc().numpy())
        counter = 0
        while solver.successful():
            solver.integrate(tmax, step=True)
            fluxes = self.builder.fluxes().detach().numpy()[:-1, -1]
            xarr.append(solver.y)
            tarr.append(solver.t)
            logc.append(self.builder.get_logc().numpy())
            fluxarr.append(fluxes)
            if solver.t > tmax:
                break
            counter += 1
            if print_frequency and (counter%print_frequency) == 0:
                print(f"{100*solver.t/tmax}%")
        self.x = np.stack(xarr, axis=0)
        self.t = np.array(tarr)
        self.logc = np.stack(logc, axis=0)
        self.fluxes = np.stack(fluxarr, axis=0)
        
    def set_initial_guess(self):
        self.builder.set_initial_guess_from_bulk()
        _ = self.builder.solve_lma(simplified=True);
        _ = self.builder.solve_lma(simplified=False);

    def f(self, t : float, y : np.ndarray):
        #y : mol/kg H2O
        self.builder.cbulk = {el:y[i] for i, el
                              in enumerate(self.builder.eqsys.solute_elements)}
        _ = self.builder.solve_lma(simplified=False);
        fluxes = self.builder.fluxes().detach().numpy()[:-1, -1] #mol/m2
        dy_molm3 = fluxes*4/self.pipe_diameter #mol/m3
        dy_molal = dy_molm3/self.water_density
        return dy_molal

    def load(self, filename : str):
        if not filename.endswith(".npz"):
            filename = filename + ".npz"
        with np.load(filename) as f:
            self.x = f["x"]
            self.t = f["t"]
            self.logc = f["logc"]
            self.fluxes = f["fluxes"]
    
    def save(self, filename : str):
        np.savez(filename, t=self.t, x=self.x,
                 logc=self.logc, fluxes=self.fluxes)


def reynolds_number(flow_velocity : float,
                    pipe_diameter : float,
                    TK : float = 298.15,
                    kinematic_viscosity : Optional[float] = None): #Dimensionless
    """
        Calculates Reynolds number of water from velocity and diameter
    """
    kinematic_viscosity = kinematic_viscosity or pyequion2.water_properties.water_kinematic_viscosity(TK)
    return flow_velocity*pipe_diameter/kinematic_viscosity


def darcy_friction_factor(flow_velocity : float,
                          pipe_diameter : float,
                          TK : float = 298.15,
                          kinematic_viscosity : Optional[float] = None):
    reynolds = reynolds_number(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    if reynolds < 2300:
        return 64/reynolds
    else: #Blasius
        return 0.316*reynolds**(-1./4)
    

def get_shear_velocity(flow_velocity : float,
                       pipe_diameter : float,
                       TK : float = 298.15,
                       kinematic_viscosity : Optional[float] = None):
    f = darcy_friction_factor(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return np.sqrt(f/8.0)*flow_velocity
        