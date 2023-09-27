"""
© 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.
"""

from pickle import load

from numpy import array, empty

from reactor_modeling import (thermo_c_ideal, spec_names, spec_molar_masses, mix_molar_mass,
                              rxn_rate_Klugherz, calc_spec_rates, pore_eff_thiele1,
                              cat_geom_sphere, trans_Re, trans_Pr, trans_Sc,
                              turb_Nu_text, film_flux_x, rxr_plugflow_x)
from parameter_set import ParameterSet, get_default_params, set_default_params

# Import the experimental results from their pickle file:
file_name = 'expt_integrated.p'
with open(file_name, 'rb') as file_obj:
    expt_data = load(file_obj)
n_dsgn = len(expt_data)  # number of experiments in the design

# Run the model at each of the same conditions:
# Submodel selections
thermo_c = thermo_c_ideal
rxn_rate = rxn_rate_Klugherz
pore_eff = pore_eff_thiele1
cat_geom = cat_geom_sphere
turb_Nu = turb_Nu_text

# Parameter values
PSS1= type('ParameterSetSubclass1', (ParameterSet,), dict())
θ = PSS1(# Thermodynamic properties:
        thermo=dict(P=(2 * 101.325e3),  # pressure [Pa]
                    T=(200 + 273.15),  # Temperature [K]
                    **get_default_params(thermo_c_ideal)),
        # Chemical species properties:
        spec=dict(**get_default_params(spec_molar_masses),
                    D = array([2.0e-5, 2.6e-5, 1.7e-5, 2.1e-5, 3.3e-5, 2.4e-5]),  # diff. coefs.
                    μ_visc=2.6e-5),  # viscosity [Pa s = kg / (s m)]
        # Chemical reaction properties & parameters:
        rxn=get_default_params(rxn_rate),
        # Turbulence parameters:
        turb_bed=get_default_params(turb_Nu),
        # Catalyst properties:
        cat=dict(ρ_cat=881,  # catalyst density [kg / m^3]
                 a_cat=0.8e3,  # catalyst surface area (including pores) per mass [m^2/kg]
                 ε_void=0.75,  # as-packed void fraction (gaps-between pellets) [-]
                 cat_geom=dict(D_pellet=2e-2),  # pellet diameter [m]
                 D_pore=4e-6,  # diameter of the pores [m]
                 # D_pore = 40e-9,  # diameter of the pores [m]  # WARNING: for testing only!
                 L_pore=1e-3),  # characteristic pore length [m],
        # Reactor properties & operation:
        rxr=dict(l_rxr=12.8,  # reactor length (industrial reactor) [m]
                 # l_rxr=2.0,  # reactor length (laboratory reactor) [m]
                 r_rxr=1.956e-2))

# Run the coupled model at each of the experimental conditions:
n_z = 600
x_model = empty((n_dsgn, 6))
N_model = empty(n_dsgn)
for i in range(n_dsgn):
    print(i)
    θ.T = expt_data[i]['scenario']['T']
    out = rxr_plugflow_x(expt_data[i]['scenario']['x'], expt_data[i]['scenario']['u'],
                         thermo_c, rxn_rate, pore_eff, cat_geom, turb_Nu, θ, n_z)
    Zrxr, Nrxr, Xrxr, Ncat, Xcat, ε_film, ε_pore = out
    x_model[i] = Xrxr[-1]
    N_model[i] = Nrxr[-1].sum()

print(x_model)

# Now, x_model[i] & N_model[i] can be compared directly to
# expt_data[i]['outcome']['x'] & expt_data[i]['outcome']['N'].