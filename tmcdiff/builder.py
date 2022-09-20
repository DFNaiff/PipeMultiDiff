# -*- coding: utf-8 -*-
from typing import List

import jax.numpy as jnp
import scipy.linalg
import jax
import scipy.optimize

import pyequion2


class FixedGridTransportBuilder(object):
    def __init__(self, eqsys, TK, shear_velocity, kinematic_viscosity, cbulk, phases,
                 reaction_constant = "inf"):
        self.eqsys = eqsys
        self.TK = TK
        self.shear_velocity = shear_velocity
        self.kinematic_viscosity = kinematic_viscosity
        self.cbulk = cbulk
        self.phases = phases
        self.reaction_constant = reaction_constant
        self.simplified = False
        
    def set_species(self, species_to_remove=[]):
        self.basedict = {k: v
                         for k, v in pyequion2.datamods.chemical_potentials.items()
                         if k in self.eqsys.solutes
                         and k not in species_to_remove}
        self.species = [spec for spec in self.eqsys.species[1:] if spec not in species_to_remove]
        self.species_ind = jnp.array([self.eqsys.species.index(spec) for spec in self.species])
        self.solid_ind = jnp.array([self.eqsys.solid_phase_names.index(phase) for phase in self.phases])
        self.reduced_formula_matrix = self.eqsys.formula_matrix[2:, self.species_ind]
        self.reduced_reaction_vector = \
            self.eqsys.solid_stoich_matrix[self.solid_ind, :][:, self.species_ind]
        self.closure_matrix = jnp.array(
            scipy.linalg.null_space(
                (self.reduced_formula_matrix@(self.reduced_reaction_vector.T)).T
            ).T
        )
        self.nspecies = len(self.species)
        self.logk_solid = self.eqsys.get_solid_log_equilibrium_constants(self.TK)[self.solid_ind]
        self.reduced_standard_potentials = self.eqsys.get_standard_chemical_potentials(self.TK)[self.species_ind]
        self.reduced_standard_potentials /= (pyequion2.constants.GAS_CONSTANT*self.TK)    
        self.reduce_activity_function()
        self.reduced_diffusion_coefficients = pyequion2.equilibrium_backend.diffusion_coefficients.get_diffusion_coefficients(
                                        self.species, self.TK)
        self.reduced_diffusion_coefficients = jnp.array(self.reduced_diffusion_coefficients)/self.kinematic_viscosity
        
    def set_initial_guess(self):
        self.logc = jnp.ones([self.nspecies, self.ngrid])*0.0 - 1.0
        self.mu = self.reduced_standard_potentials[..., None] + self.logc
        self.logcmu = jnp.vstack([self.logc, self.mu])
        self.xinit = self.logcmu.flatten()
        
    def reduce_activity_function(self):
        activity_model = self.eqsys.activity_model
        self._actfunc = pyequion2.equilibrium_backend.ACTIVITY_MODEL_MAP[activity_model](
                            self.species, backend="jax")
        
    def activity_model_func(self, molals, TK):
        return self._actfunc(molals, TK)[..., 1:]/pyequion2.constants.LOG10E
        
    def make_grid(self, ngrid, ymax):
        self.ngrid = ngrid
        self.npoints = ngrid + 1
        self.ymax = ymax
        self.ypoints, self.ystep = jnp.linspace(0, self.ymax, self.npoints, retstep=True)
        self.ygrid = self.ypoints[:-1] + self.ystep/2
        
    def wall_diffusion_plus(self, yplus):
        return 9.5*1e-4*yplus**3
    
    def wall_diffusion_plus_deriv(self, yplus):
        return 3*9.5*1e-4*yplus**2

    def transport_residual(self, logc, mu):
        #logc : (nsolutes, ngrid)
        #npoints = ngrid + 1
        logg = self.activity_model_func(jnp.exp(logc).T, self.TK).T #(nsolutes, ngrid)
        c = jnp.exp(logc) #(nsolutes, ngrid)
        loga = logg + logc #(nsolutes, ngrid)
        ypoints = self.ypoints #(npoints,)
        ystep = self.ystep
        mdiffs = self.reduced_diffusion_coefficients[..., None] #(nsolutes, 1)
        tdiffs = self.wall_diffusion_plus(ypoints)[1:-1] #(ngrid-1,)
        A = self.reduced_formula_matrix #(nels, nsolutes)
        C = self.closure_matrix #(nels-nreac, nsolutes)
        R = self.reduced_reaction_vector #(nreac, nsolutes)
        logKsp = self.logk_solid[..., None] #(nreac, 1)
        cbulk = self.cbulk[..., None] #(nels, 1)
        cinterp = (c[:, 1:] + c[:, :-1])*0.5 #(nsolutes, ngrid-1)
        dmu = (loga[:, 1:] - loga[:, :-1])/ystep #(nsolutes, ngrid-1)
        dc = (c[:, 1:] - c[:, :-1])/ystep #(nsolutes, ngrid-1)
        if self.simplified:
            jmiddle = -A@(mdiffs*dc + tdiffs*dc) #(nsolutes, ngrid-1)
        else:
            jmiddle = -A@(mdiffs*cinterp*dmu + tdiffs*dc) #(nsolutes, ngrid-1)
        jleft = jmiddle[:, :1] #(nsolutes, 1)
        cright = c[:, -1:] #(nsolutes, 1)
        logaleft = loga[:, :1] #(nsolutes, 1)
        middle_residual = jmiddle[:, 1:] - jmiddle[:, :-1] #(nsolutes, ngrid-2)
        right_residual = A@cright - cbulk #(nsolutes, 1)
        if self.reaction_constant == 'inf':
            upper_left_residual = R@logaleft - logKsp #(nreac, 1)
            lower_left_residual = C@jleft #(nels - nreac, 1)
            left_residual = jnp.vstack([upper_left_residual, lower_left_residual]) #(nsolutes, 1)
        else:
            logsatur = R@logaleft - logKsp #nreac - 1 #(nreac, 1)
            jsatur = self.reaction_constant*jnp.clip(jnp.exp(logsatur) - 1, 0, jnp.inf) #(nreac, 1)
            left_residual = -A@(R.T)@jsatur - jleft #(nels, 1)
            #self.reduced_reaction_vector
            pass
        residual = jnp.hstack([left_residual, middle_residual, right_residual]) #(nsolutes, ngrid)
        return residual
    
    def fluxes(self):
        logg = self.activity_model_func(jnp.exp(self.logc).T, self.TK).T #(nsolutes, ngrid)
        c = jnp.exp(self.logc) #(nsolutes, ngrid)
        loga = logg + self.logc #(nsolutes, ngrid)
        ypoints = self.ypoints #(npoints,)
        ystep = self.ystep
        mdiffs = self.reduced_diffusion_coefficients[..., None] #(nsolutes, 1)
        tdiffs = self.wall_diffusion_plus(ypoints)[1:-1] #(ngrid-1,)
        A = self.reduced_formula_matrix #(nels, nsolutes)
        cinterp = (c[:, 1:] + c[:, :-1])*0.5 #(nsolutes, ngrid-1)
        dmu = (loga[:, 1:] - loga[:, :-1])/ystep #(nsolutes, ngrid-1)
        dc = (c[:, 1:] - c[:, :-1])/ystep #(nsolutes, ngrid-1)
        if self.simplified:
            jmiddle = -A@(mdiffs*dc + tdiffs*dc) #(nels, ngrid-1)
        else:
            jmiddle = -A@(mdiffs*cinterp*dmu + tdiffs*dc) #(nels, ngrid-1)
        return jmiddle
    
    def potential_residual(self, logc, mu):
        #logc : (nsolutes, ngrid)
        #mu : (nsolutes, ngrid)
        print(self.TK)
        logg = self.activity_model_func(jnp.exp(logc).T, self.TK).T
        loga = logg + logc
        mu0 = self.reduced_standard_potentials[..., None]
        return mu - (mu0 + loga)
    
    def full_residual(self, logcmu):
        #logcmu : (2*nsolutes, ngrid)
        n = logcmu.shape[0]
        logc = logcmu[:n//2, :]
        mu = logcmu[n//2:, :]
        res1 = self.transport_residual(logc, mu) #(nels, ngrid)
        res3 = self.potential_residual(logc, mu) #(nsolutes, ngrid)
        res = jnp.vstack([res1, res3]) #(nels + nsolutes, ngrid)
        return res
    
    def bulk_residual(self, logcmu):
        n = logcmu.shape[0]
        logc = logcmu[:n//2, :]
        mu = logcmu[n//2:, :]        
        c = jnp.exp(logc) #(nsolutes, 1)
        A = self.reduced_formula_matrix #(nels, nsolutes)
        cbulk = self.cbulk[..., None] #(nels, 1)]
        res1 = A@c - cbulk
        res3 = self.potential_residual(logc, mu) #(nsolutes, 1)
        res = jnp.vstack([res1, res3]) #(nels + nsolutes, ngrid)
        return res
    
    def gibbs_free_energy(self, logcmu):
        n = logcmu.shape[0]
        logc = logcmu[:n//2, :]
        mu = logcmu[n//2:, :]
        c = jnp.exp(logc)
        return jnp.mean(jnp.sum(c*mu, axis=0))    
    
    def flattened_equality_constraint(self, x):
        cmu = x.reshape(2*self.nspecies, self.ngrid)
        return self.full_residual(cmu).flatten()
    
    def flattened_minimization_objective(self, x):
        cmu = x.reshape(2*self.nspecies, self.ngrid)
        return self.gibbs_free_energy(cmu).flatten()[0]
        
    def flattened_bulk_constraint(self, x):
        return self.bulk_residual(x.reshape(2*self.nspecies, 1)).flatten()
    
    def flattened_bulk_minimization_objective(self, x):
        return self.gibbs_free_energy(x.reshape(2*self.nspecies, 1)).flatten()[0]
        
    def wall_length(self):
        return self.kinematic_viscosity/self.shear_velocity
    
    def wall_time(self):
        return self.kinematic_viscosity/(self.shear_velocity**2)
    
    def get_log_equilibrium_constants(self, TK):
        return self.eqsys.get_log_equilibrium_constants(TK)
    
    def solve(self, simpleonly=False, warmup=True):
        equality_constraint = scipy.optimize.NonlinearConstraint(
            jax.jit(self.flattened_equality_constraint),
            lb=0.0,
            ub=0.0,
            jac = jax.jit(jax.jacfwd(self.flattened_equality_constraint)))
        if warmup or simpleonly:
            self.simplify()
            sol_simple = scipy.optimize.minimize(
                                jax.jit(self.flattened_minimization_objective),
                                self.xinit,
                                jac = jax.jit(jax.grad(self.flattened_minimization_objective)),
                                constraints=equality_constraint,
                                method='trust-constr')
        if simpleonly:
            sol = sol_simple
        else:
            self.complexify()
            if warmup:
                xinit = sol_simple.x
            else:
                xinit = self.xinit
            sol = scipy.optimize.minimize(jax.jit(self.flattened_minimization_objective),
                                          xinit,
                                          jac = jax.jit(jax.grad(self.flattened_minimization_objective)),
                                          constraints=equality_constraint,
                                          method='trust-constr')
        self.xinit = sol.x
        self.logcmu = sol.x.reshape(self.nspecies*2, self.ngrid)
        self.logc, self.mu = self.logcmu[:self.nspecies, :], self.logcmu[self.nspecies:, :]
        return sol, sol_simple
    
    def solve_bulk(self):
        xinit = self.logcmu[:, -1][..., None]
        equality_constraint = scipy.optimize.NonlinearConstraint(
            jax.jit(self.flattened_bulk_constraint),
            lb=0.0,
            ub=0.0,
            jac = jax.jit(jax.jacfwd(self.flattened_bulk_constraint)))
        sol = scipy.optimize.minimize(jax.jit(self.flattened_bulk_minimization_objective),
                                      xinit,
                                      jac=jax.jit(jax.jacfwd(jax.jit(self.flattened_bulk_minimization_objective))),
                                      constraints=equality_constraint,
                                      method='trust_constr')
        return sol.x
        
    def simplify(self):
        self.simplified = True
    
    def complexify(self):
        self.simplified = False