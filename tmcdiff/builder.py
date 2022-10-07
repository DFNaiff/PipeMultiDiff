from typing import Dict, List, Union, Optional, TypeVar

import numpy as np
import torch
import functorch
import scipy.linalg
import pyequion2


GenericTensor = TypeVar("GenericTensor")


class TransportBuilder(object):
    def __init__(self, eqsys : pyequion2.EquilibriumBackend,
                 TK : float,
                 shear_velocity : float,
                 cbulk : Dict[str, float],
                 phases : List[str],
                 kreaction : Optional[Union[float, GenericTensor]] = None,
                 flux_residual_tol : float = 1e0):
        self.eqsys = eqsys
        self.TK = TK
        self.shear_velocity = shear_velocity
        self.kinematic_viscosity = pyequion2.water_properties.water_kinematic_viscosity(self.TK)
        self.water_density = pyequion2.water_properties.water_density(self.TK)
        self.cbulk = cbulk
        self.phases = phases
        self.kreaction = kreaction
        self.bturb = pyequion2.constants.TURBULENT_VISCOSITY_CONSTANT
        self.flux_residual_tol = flux_residual_tol
        self.nsolid = len(phases)
        self.simplified = False
        self.idealized = False
        self.species_to_remove = []
        self.base_dict = None
        self.species = None
        self.species_ind = None
        self.solid_ind = None
        self.reduced_formula_matrix = None
        self.reduced_stoich_matrix = None
        self.logk = None
        self.reduced_reaction_vector = None
        self.closure_matrix = None
        self.nspecies = None
        self.logk_solid = None
        self.reduced_standard_potentials = None
        self.reduced_diffusion_coefficients = None
        self.ngrid = None
        self.npoints = None
        self.ymax = None
        self.ypoints = None
        self.ystep = None
        self.ygrid = None
        self.x = None
        self.warmup_factor = 1.0
        
    def get_logc(self):
        logc = self.x.reshape(self.nspecies, self.ngrid)
        return logc

    def set_species(self, species_to_remove : Optional[List[str]] = None):
        if species_to_remove is not None:
            self.species_to_remove = species_to_remove
        self.basedict = {k: v
                         for k, v in pyequion2.datamods.chemical_potentials.items()
                         if k in self.eqsys.solutes
                         and k not in self.species_to_remove}
        self.species = [spec for spec in self.eqsys.species[1:] if spec not in self.species_to_remove]
        self.species_ind = [self.eqsys.species.index(spec) for spec in self.species]
        self.solid_ind = [self.eqsys.solid_phase_names.index(phase) for phase in self.phases]
        self.reduced_formula_matrix = self.eqsys.formula_matrix[2:, self.species_ind]
        self.reduced_stoich_matrix = self.eqsys.stoich_matrix[:, self.species_ind]
        self.logk = self.eqsys.get_log_equilibrium_constants(self.TK)
        self.reduced_reaction_vector = \
            self.eqsys.solid_stoich_matrix[self.solid_ind, :][:, self.species_ind]
        self.closure_matrix = torch.tensor(
            scipy.linalg.null_space(
                (self.reduced_formula_matrix@(self.reduced_reaction_vector.T)).T
            ).T
        )
        if self.kreaction == "inf":
            self.kreaction = None
        if self.kreaction is not None:
            self.kreaction_ = (torch.tensor(self.kreaction, dtype=torch.float)
                               if type(self.kreaction) != torch.Tensor
                               else self.kreaction)
            self.kreaction_ = (torch.ones([self.nsolid])*self.kreaction)[..., None]
        else:
            self.kreaction_ = None
        self.nspecies = len(self.species)
        self.logk_solid = self.eqsys.get_solid_log_equilibrium_constants(self.TK)[self.solid_ind]
        
        self.reduced_standard_potentials = self.eqsys.get_standard_chemical_potentials(self.TK)[self.species_ind]
        self.reduced_standard_potentials /= (pyequion2.constants.GAS_CONSTANT*self.TK)    
        self.reduce_activity_function()
        self.reduced_diffusion_coefficients = pyequion2.equilibrium_backend.diffusion_coefficients.get_diffusion_coefficients(
                                        self.species, self.TK)
        self.reduced_diffusion_coefficients = torch.tensor(self.reduced_diffusion_coefficients,
                                                           dtype=torch.float)
        self.reduced_diffusion_coefficients /=self.kinematic_viscosity
        self.mean_diffusion_coefficients = np.median(self.reduced_diffusion_coefficients)
        self.water_reaction_index = [i for i, reac in enumerate(self.eqsys.reactions)
                                     if is_water_reaction(reac)]
        
    def change_temperature(self, TK : float):
        self.TK = TK
        self.kinematic_viscosity = pyequion2.water_properties.water_kinematic_viscosity(self.TK)
        self.water_density = pyequion2.water_properties.water_density(self.TK)
        self.set_species()
        
    def reduce_activity_function(self):
        activity_model = self.eqsys.activity_model
        self._actfunc = pyequion2.equilibrium_backend.ACTIVITY_MODEL_MAP[activity_model](
                            self.species, backend="torch")
        
    def activity_model_func(self, molals : torch.Tensor,
                            TK : float):
        if self.idealized:
            return torch.zeros_like(molals)
        else:
            return self._actfunc(molals, TK)[..., 1:]/pyequion2.constants.LOG10E
        
    def make_grid(self, ngrid : int, ymax : float):
        self.ngrid = ngrid
        self.npoints = ngrid + 1
        self.ymax = ymax
        self.ypoints = torch.linspace(0, self.ymax, self.npoints)
        self.ystep = self.ypoints[1] - self.ypoints[0]
        self.ygrid = self.ypoints[:-1] + self.ystep/2
        
    def wall_diffusion_plus(self, yplus : torch.Tensor):
        return self.bturb*yplus**3

    def wall_length(self):
        return self.kinematic_viscosity/self.shear_velocity
    
    def get_log_equilibrium_constants(self, TK : float):
        return self.eqsys.get_log_equilibrium_constants(TK)
    
    def make_bulk_vector(self):
        base = [self.cbulk[el] for el in self.eqsys.solute_elements] + [0.0]
        base = torch.tensor(base, dtype=torch.float)
        base = base[..., None]
        return base
    
    def transport_residual(self, logc : Optional[torch.Tensor] = None,
                           c : Optional[torch.Tensor] = None):
        #logc : (nsolutes, ngrid)
        #npoints = ngrid + 1
        assert (c is not None or logc is not None)
        if c is None:
            c = torch.exp(logc) #(nsolutes, ngrid)
        else:
            logc = torch.log(c)
        logg = self.activity_model_func(torch.exp(logc).T, self.TK).T #(nsolutes, ngrid)
        loga = logg + logc #(nsolutes, ngrid)
        ypoints = self.ypoints #(npoints,)
        ystep = self.ystep
        mdiffs = self.get_reduced_diffusion_coefficients() #(nsolutes, 1)
        tdiffs = self.wall_diffusion_plus(ypoints)[1:-1] #(ngrid-1,)
        A = self.reduced_formula_matrix #(nels, nsolutes)
        C = self.closure_matrix #(nnoncon-nreac, nsolutes)
        R = self.reduced_reaction_vector #(nreac, nsolutes)
        logKsp = self.logk_solid[..., None] #(nreac, 1)
        cbulk = self.make_bulk_vector() #(nels, 1) #(mol/kgH2O)
        cinterp = (c[:, 1:] + c[:, :-1])*0.5 #(nsolutes, ngrid-1)
        dmu = (loga[:, 1:] - loga[:, :-1])/ystep #(nsolutes, ngrid-1)
        dc = (c[:, 1:] - c[:, :-1])/ystep #(nsolutes, ngrid-1)
        if self.simplified:
            jmiddle = -A@(mdiffs*dc + tdiffs*dc)*self.water_density #(nels, ngrid-1) #(mol/(m3))
        else:
            jmiddle = -A@(mdiffs*cinterp*dmu + tdiffs*dc)*self.water_density #(nels, ngrid-1)
        jleft = jmiddle[:, :1] #(nels, 1)
        cright = c[:, -1:] #(nsolutes, 1)
        logaleft = loga[:, :1] #(nsolutes, 1)
        middle_residual = (jmiddle[:, 1:] - jmiddle[:, :-1])/ystep #(nels, ngrid-2)
        right_residual = A@cright - cbulk #(nsolutes, 1) #mol/l
        if self.kreaction is None:
            upper_left_residual = R@logaleft - logKsp #(nreac, 1)
            lower_left_residual = C@(jleft) #(nnoncon - nreac, 1) #(mol/(m3))
            left_residual = torch.cat([upper_left_residual, lower_left_residual], dim=0) #(nsolutes, 1)
        else:
            logsatur = R@logaleft - logKsp #(nreac, 1)
            jsatur = -self.kreaction*torch.relu(torch.exp(logsatur) - 1) #(nreac, 1) #(mol/m3)
            #jsatur = -self.kreaction*logsatur
            jend = A@(R.T)@jsatur #(nels, 1)
            left_residual = (jend - jleft)/ystep #(nels, 1)
        residual = torch.cat([left_residual, middle_residual, right_residual], dim=1) #(nsolutes, ngrid)
        return residual
    
    def fluxes(self, logc : Optional[torch.Tensor] = None,
               c : Optional[torch.Tensor] = None):
        #logc : (nsolutes, ngrid)
        #npoints = ngrid + 1
        if logc is None and c is None:
            logc = self.get_logc()
        assert (c is not None or logc is not None)
        if c is None:
            c = torch.exp(logc) #(nsolutes, ngrid)
        else:
            logc = torch.log(c)
        logg = self.activity_model_func(torch.exp(logc).T, self.TK).T #(nsolutes, ngrid)
        loga = logg + logc #(nsolutes, ngrid)
        ypoints = self.ypoints #(npoints,)
        ystep = self.ystep
        mdiffs = self.get_reduced_diffusion_coefficients() #(nsolutes, 1)
        tdiffs = self.wall_diffusion_plus(ypoints)[1:-1] #(ngrid-1,)
        A = self.reduced_formula_matrix #(nels, nsolutes)
        R = self.reduced_reaction_vector #(nreac, nsolutes)
        logKsp = self.logk_solid[..., None] #(nreac, 1)
        cinterp = (c[:, 1:] + c[:, :-1])*0.5 #(nsolutes, ngrid-1)
        dmu = (loga[:, 1:] - loga[:, :-1])/ystep #(nsolutes, ngrid-1)
        dc = (c[:, 1:] - c[:, :-1])/ystep #(nsolutes, ngrid-1)
        if self.simplified:
            jmiddle = -A@(mdiffs*dc + tdiffs*dc)*self.water_density #(nels, ngrid-1)
        else:
            jmiddle = -A@(mdiffs*cinterp*dmu + tdiffs*dc)*self.water_density #(nels, ngrid-1)
        if self.kreaction is not None:
            logaleft = loga[:, :1]
            logsatur = R@logaleft - logKsp #(nreac, 1)
            jsatur = -self.kreaction*torch.relu(torch.exp(logsatur) - 1) #(nreac, 1) #mol/m4 #TODO: Correct unit for kreaction
            jend = A@(R.T)@jsatur #(nels, 1)
            jfull = torch.cat([jend, jmiddle], dim=-1)
        else:
            jfull = jmiddle
        jfull *= self.kinematic_viscosity/self.wall_length() #mol/m3 to mol/(m2*s)
        return jfull
        
    
    def lma_residual(self, logc : torch.Tensor = None, 
                     c : Optional[torch.Tensor] = None,
                     only_water : bool = False):
        assert (c is not None or logc is not None)
        if c is None:
            c = torch.exp(logc) #(nsolutes, ngrid)
        else:
            logc = torch.log(c)
        logg = self.activity_model_func(torch.exp(logc).T, self.TK).T
        loga = logg + logc
        if only_water:
            nu = self.reduced_stoich_matrix[self.water_reaction_index, :]
            logK = self.logk[self.water_reaction_index]
        else:
            nu = self.reduced_stoich_matrix
            logK = self.logk
        res = nu@loga - logK[..., None]
        return res
    
    def full_residual(self, logc : Optional[torch.Tensor] = None,
                      c : Optional[torch.Tensor] = None,
                      lma : bool = True,
                      include_water : bool = True):
        #logc : (2*nsolutes, ngrid)
        residual_factor = 1/self.flux_residual_tol
        res1 = self.transport_residual(logc, c) #(nels, ngrid)
        res1 *= residual_factor
        if lma or include_water:
            res2 = self.lma_residual(logc, c, only_water=not lma)
            res = torch.cat([res1, res2], dim=0)
        else:
            res = res1
        return res

    def gibbs_free_energy(self, logc : Optional[torch.Tensor] = None,
                          c : Optional[torch.Tensor] = None):
        assert (c is not None or logc is not None)
        if c is None:
            c = torch.exp(logc) #(nsolutes, ngrid)
        else:
            logc = torch.log(c)
        mu = self.chemical_potentials(logc)
        return torch.mean(torch.sum(c*mu, dim=0))
    
    def chemical_potentials(self, logc):
        logg = self.activity_model_func(torch.exp(logc).T, self.TK).T
        loga = logg + logc
        mu0 = self.reduced_standard_potentials[..., None]
        mu = mu0 + loga
        return mu

    def set_initial_guess_from_bulk(self):
        eqsysbulk = pyequion2.EquilibriumSystem(self.eqsys.solute_elements,
                                                activity_model=self.eqsys.activity_model,
                                                from_elements=True)
        bulk_solution, bulk_stats = eqsysbulk.solve_equilibrium_mixed_balance(self.TK,
                                                                              molal_balance=self.cbulk)
        bulk_molals = bulk_solution.molals
        logcvector = [np.log(bulk_molals[spec]) for spec in self.species]
        logcvector = torch.tensor(logcvector, dtype=torch.float)[:, None]
        logc = torch.zeros([self.nspecies, self.ngrid]) + logcvector
        self.x = logc.flatten()
        lambd0 = 1/max(self.cbulk.values())
        lambd_ = torch.zeros([len(self.cbulk) + 2, self.ngrid]) + lambd0
        self.lambd = lambd_.flatten()
        
    def set_initial_guess_lagrangian(self, method='lm'):
        raise NotImplementedError
        oldngrid = self.ngrid
        self.make_grid(2, self.ymax)
        self.set_initial_guess_from_bulk()
        _ = self.solve_lma(simplified=True);
        _ = self.solve_lma(simplified=False);
        solgem = self.solve_gem_lagrangian(simplified=False, method=method)
        x = self.x
        lambd_ = self.lambd
        logc = x.reshape([self.nspecies, self.ngrid])
        lambd = lambd_.reshape([len(self.cbulk) + 2, self.ngrid])
        self.make_grid(oldngrid, self.ymax)
        logc = torch_linspace(logc[:, 0], logc[:, 1], self.ngrid).T
        lambd = torch_linspace(lambd[:, 0], lambd[:, 1], self.ngrid).T
        self.x = logc.flatten()
        self.lambd = lambd.flatten()
        return solgem
        
    def solve_gem_lagrangian(self, simplified : bool = False,
                             method='lm', options=None):
        options = dict() if options is None else options
        self.simplified = simplified
        def flat_residual(x):
            logc = x.reshape([self.nspecies, self.ngrid])
            res = self.full_residual(logc, lma=False)
            return res.flatten()
        
        def flat_objective(x):
            logc = x.reshape([self.nspecies, self.ngrid])
            res = self.gibbs_free_energy(logc)
            return res.flatten()[0]
        f_t = flat_objective
        df_t = functorch.jacrev(f_t)
        g_t = flat_residual
        dg_t = functorch.jacrev(g_t)
        df = torch_wrap(df_t)
        dg = torch_wrap(dg_t)
        x0 = self.x.detach().numpy()
        blambd = df(x0)
        Alambd = dg(x0).T
        lambd0, _, _, _ = np.linalg.lstsq(Alambd, blambd, rcond=None)
        xlambd0 = np.concatenate([x0, lambd0], axis=0)
        h_t = make_lagrangian_residual_torch(g_t,
                                             df_t,
                                             dg_t,
                                             len(self.x),
                                             len(self.lambd))
        loss_t = lambda x : torch.sum(h_t(x)**2)
        dloss_t = functorch.grad(loss_t)
        d2loss_t = functorch.hessian(loss_t)
        loss = torch_wrap(loss_t)
        dloss = torch_wrap(dloss_t)
        d2loss = torch_wrap(d2loss_t)
        sol = scipy.optimize.minimize(loss, xlambd0, jac=dloss, method='TNC',
                                      hess=d2loss) #TNC
        xlambd0 = sol.x
        sol = scipy.optimize.minimize(loss, xlambd0, jac=dloss, method='trust-constr',
                                      hess=d2loss, options=options) #TNC
        x, lambd = sol.x[:len(self.x)], sol.x[len(self.x):]
        x = torch.tensor(x, dtype=torch.float)
        lambd = torch.tensor(lambd, dtype=torch.float)
        self.x = x
        self.lambd = lambd
        return sol
        
    def solve_lma(self, simplified : bool = False):
        self.simplified = simplified
        def flat_residual(x):
            logc = x.reshape([self.nspecies, self.ngrid])
            res = self.full_residual(logc, lma=True)
            return res.flatten()
        f = torch_wrap(flat_residual)
        jac = torch_wrap(functorch.jacrev(flat_residual))
        x0 = self.x.detach().numpy()
        sol = scipy.optimize.root(f, x0, jac=jac, method="lm", tol=1e-12)
        x = torch.tensor(sol.x, dtype=torch.float)
        self.x = x
        return sol
    
    def get_reduced_diffusion_coefficients(self):
        d0 = self.mean_diffusion_coefficients
        d1 = self.reduced_diffusion_coefficients
        d = (1 - self.warmup_factor)*d0 + self.warmup_factor*d1
        return d[..., None]
    
    @property
    def xlambd(self):
        return torch.cat([self.x, self.lambd], dim=0)
    
    @property
    def nx(self):
        return len(self.x)
    
    @property
    def nlambd(self):
        return len(self.lambd)


def make_lagrangian_residual_jacobian(g, df, dg, d2f, d2g, nx, nlambd):
    def residual(xlambd):
        x, lambd = xlambd[..., :nx], xlambd[..., nx:]
#        print(x.shape)
#        print(lambd.shape)
#        print(df(x).shape)
#        print(dg(x).shape)
        res1 = df(x) - (dg(x).T)@lambd
        res2 = g(x)
        res = np.concatenate([res1, res2], axis=-1)
        return res
    def jacobian(xlambd):
        x, lambd = xlambd[..., :nx], xlambd[..., nx:]
        jac11 = d2f(x) - d2g(x, lambd)
        jac12 = -dg(x).T
        jac21 = dg(x)
        jac22 = np.zeros([nlambd, nlambd])
        jac = np.block([[jac11, jac12], [jac21, jac22]])
        return jac
    return residual, jacobian


def make_lagrangian_residual_torch(g, df, dg, nx, nlambd):
    def residual(xlambd):
        x, lambd = xlambd[..., :nx], xlambd[..., nx:]
        res1 = df(x) - (dg(x).T)@lambd
        res2 = g(x)
        res = torch.cat([res1, res2], axis=-1)
        return res
    return residual


def torch_wrap(f):
    def g(*args):
        args = [torch.tensor(arg, dtype=torch.float) for arg in args]
        res = f(*args).detach().numpy()
        return res
    return g


def is_water_reaction(reac):
    return 'H2O' in reac and 'OH-' in reac and 'H+' in reac


@torch.jit.script
def torch_linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out