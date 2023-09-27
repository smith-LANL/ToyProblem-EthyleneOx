"""
This contains the basic models for the simulation of a steady-state ethylene reactor.

© 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.

@author: Sean T. Smith, 1 Nov. 2022
"""
from numpy import array, empty, linspace, maximum, sqrt, exp, log10, tanh, inf, pi as π
from scipy.optimize import least_squares#, fsolve
from parameter_set import set_default_params

# Thermodynamic model (EOS) for molar concentration:
def thermo_c_ideal(P, T, R=8.31446):
    """
    Calculate the concentration [mol/m^3] from pressure [Pa] and temperature [K] using the
    ideal-gas law EOS.
    Parameter notes:
    R — gas constant [m^3 Pa / (mol K) = kg m^2 / (s^2 mol K)]
    """
    return P / (R * T)  # molar concentration [mol/m^3]

# Chemical species properties:
spec_names = ['C$_2$H$_4$', 'O$_2$', 'C$_2$H$_4$O', 'CO$_2$', 'H$_2$O', 'N$_2$']
def spec_molar_masses(M_atom=(1e-3 * array([12.01, 1.01, 16.00, 14.01]))):
    """
    Calculate chemical species molar masses [kg / mol], in order (), from atomic mass of each
    element [kg / mol], in order (C, H, O, N), and molecular stoichiometry, n_spec x n_el.
    """
    n_el = 4  # number of atomic elements
    n_spec = 6  # number of chemical species
    # molecular stoichiometry:
    ν_spec=array([[2, 4, 0, 0],  # ethylene
                  [0, 0, 2, 0],  # oxygen
                  [2, 4, 1, 0],  # ethylene oxide
                  [1, 0, 2, 0],  # carbon dioxide
                  [0, 2, 1, 0],  # water
                  [0, 0, 0, 2]]) # nitrogen    
    M = ν_spec @ M_atom
    return M
spec_molar_masses.n_spec = 6  # number of chemical species (provide external access)

def mix_molar_mass(x, M_atom):
    """
    Calculate the mixture molar mass [kg / mol] from mole fractions and atomic mass [kg / mol].
    """
    M = spec_molar_masses(M_atom)
    M_mix = x @ M
    return M_mix

# Chemical reaction-rate models (rate-laws):
r01 = (6 + 7) / 2 * 1e-6  # correct de-normalization to satisfy the reference
r02 = (4 + 6) / 2 * 1e-6  # correct de-normalization to satisfy the reference
def rxn_rate_Klugherz(x, P, T, a_expt=0.5, r01=r01, rKo1=6.32, Ke1=0.053,
                      Ko1=0.0199, Cp1=0.0106, r02=r02, rKo2=3.57, Ke2=0.103,
                      Ko2=0.0390, Cp2=0.0080):
    """
    Calculate the reaction rates [mol-rxn. / (s m^2-cat.)] from species mole fraction [-],
    pressure [Pa] and temperature [K] using the rate law provided in Klugherz & Harriott (1971).
    Parameter notes:
    a_expt — the catalyst specific surface area as reported in the expt [m^2 / kg],
    r01 — rate coefficient for ethylene oxide production [mol. rxn. / (min. g-cat.)],
    r02 — rate coefficient for CO2/H2O production [mol. rxn. / (min. g-cat.)].
    """
    # Convert from pressure [Pa] & mole fractions [-] to partial pressures [atm].
    Pe = x[0] * P / 1.01325e5
    Po = x[1] * P / 1.01325e5
    r1 = r01 * Pe * Po**2 / (((Cp1 + Ke1 * Pe + Ko1 * Po) * (1 + rKo1 * sqrt(Po)))**2 + 1e-12)
    r2 = r02 * Pe * Po**2 / (((Cp2 + Ke2 * Pe + Ko2 * Po) * (1 + rKo2 * sqrt(Po)))**2 + 1e-12)
    rates = array([r1, r2]) / (60 * a_expt)  # unit conversion to [mol. rxn. / (s m^2-cat.)]
    return rates
# Certain parameters associated with the rxn. rate law needed elsewhere:
rxn_rate_Klugherz.n_rxn=2  # number of chemical reactions
                  # reaction stoichiometry:
rxn_rate_Klugherz.ν_rxn = array([[-1.0,-1.0],  # ethylene
                                 [-0.5,-3.0],  # oxygen
                                 [ 1.0, 0.0],  # ethylene oxide
                                 [ 0.0, 2.0],  # carbon dioxide
                                 [ 0.0, 2.0],  # water
                                 [ 0.0, 0.0]]) # nitrogen

def rxn_rate_alt(x, P, T, Ke=2.305, Ko=6.506, Cp=0.0, n1e=0.749, n1o=2.770, n2e=0.725, n2o=2.895,
                 keo=5.420e-6, kc=5.292e-6, ΔH_rxn=(-1e3 * array([1.067e2, 1.323e3]))):
    """
    Calculate the reaction rates [mol-rxn. / (s m^2-cat.)] from species mole fraction [-],
    pressure [Pa] and temperature [K] using an alternative rate law (all sites available for
    adsorption, atomic-oxygen adsorption, Eley-Rideal for eth. ox., simple-Langmuir for CO2
    variable-order rxns.).
    Parameter notes:
    keo — rate coefficient for ethylene oxide production [mol-rxn. / (s m^2-cat.)],
    kc  — rate coefficient for CO2/H2O production [mol-rxn. / (s m^2-cat.)],
    ΔH_rxn — heats of reaction [J / mol-rxn.]  (Careful: use caution with mutable default args.).
    """
    # Convert from pressure [Pa] & mole fractions [-] to partial pressures [bar].
    Pe = x[0] * P * 1e-5
    Po = x[1] * P * 1e-5
    F = 1 + Ke * Pe + sqrt(Ko * Po) + Cp + 1e-12
    θe = Pe
    θo = sqrt(Ko * Po) / F
    r1 = keo * θe**n1e * θo**n1o
    θe = Ke * Pe / F
    θo = sqrt(Ko * Po) / F
    r2 = kc * θe**n2e * θo**n2o
    rates = array([r1, r2])
    rxn_rate_alt.ΔH_rxn = ΔH_rxn  # provide access to these uncertain parameters
    return rates
# Certain parameters associated with the rxn. rate law needed elsewhere:
rxn_rate_alt.n_rxn=2  # number of chemical reactions
# reaction stoichiometry:
rxn_rate_alt.ν_rxn=array([[-1.0,-1.0],  # ethylene
                          [-0.5,-3.0],  # oxygen
                          [ 1.0, 0.0],  # ethylene oxide
                          [ 0.0, 2.0],  # carbon dioxide
                          [ 0.0, 2.0],  # water
                          [ 0.0, 0.0]]) # nitrogen

def calc_spec_rates(x, P, T, rxn_rate_law, **rate_params):
    """
    Calculate the species production rates [mol / (s m^2-cat.)] & rate of enthalpy change
    [kJ / (s m^2-cat.)] from species mole fraction [-], pressure [Pa] and temperature [K].
    """
    rxn_rate = rxn_rate_law(x, P, T, **rate_params)
    spec_rate = rxn_rate_law.ν_rxn @ rxn_rate
    if not hasattr(rxn_rate_law, 'ΔH_rxn'):
        return spec_rate, None
    else:
        heat_rate = rxn_rate_law.ΔH_rxn @ rxn_rate
        return spec_rate, heat_rate

# Models for reaction/diffusion in pores (Thiele effect):
def pore_eff_thiele1(D_pore, L_pore, Dmix, x, P, T, Ctot, rxn_rate_law, **rate_params):
    """
    Calculate the effectiveness [-] of the pore surface area for reaction by coupling the surface
    rxn. with the diffusion down the length of the pore. Linearize each reaction by its primary
    reactant in order to apply the well known Theile modulus for first-order reactions to each
    reaction independently.
    """
    n_rxn = rxn_rate_law.n_rxn
    x_norm = maximum(x[:n_rxn], 0.002)  # species & rxns. happen to be ordered by primary reactant
    rate = empty(n_rxn)
    x_mod = x.copy()
    for i in range(n_rxn):
        x_mod[i] = x_norm[i]  # TODO: Is it worth worrying about the normalization?
        rate[i] = rxn_rate_law(x_mod, P, T, **rate_params)[i]
        x_mod[i] = x[i]
    ke_lin = rate / (Ctot * x_norm)  # linearized rate coefficient
    λ = sqrt((2 * ke_lin) / (Dmix[2] * D_pore / 2))
    ε_pore = tanh(λ * L_pore) / (λ * L_pore)
    # ε_pore = array([1] * n_rxn)  # for testing!
    return ε_pore

# Models for the geometry of the catalyst pellets:
def cat_geom_sphere(D):
    """
    Calculate the surface-area-to-volume ratio [1 / m] and sphere-equivalent diameter (diameter
    for a sphere of equal area) [m] for a sphere given its diameter.
    """
    A2V = 6 / D
    Dse = D
    return A2V, Dse

def cat_geom_cylinder(H, D):
    """
    Calculate the surface-area-to-volume ratio [1 / m] and sphere-equivalent diameter (diameter
    for a sphere of equal surface area) [m] for a cylinder given its diameter & height.
    """
    A2V = 2 * (2 / D + 1 / H)
    Dse = sqrt(D * (D / 2 + H))
    return A2V, Dse

# Transport & Turbulence models for the Sherwood & Nusselt numbers:
def trans_Re(ρ, u, L, μ):
    """
    Calculate the Reynolds number [-] from fluid density [kg/m^3], velocity [m/s], characteristic
    length [m], and viscosity [kg/(s m)].
    """
    # u * L / ν
    return ρ * u * L / μ

def trans_Pr(Cp, μ, k):
    """
    Calculate the Prandtl number [-] from fluid heat capacity [J/(kg K)], viscosity [kg/(s m)] &
    thermal conductivity [J/(s m K)].
    """
    # ν / α  # kinematic viscosity [m^2/s] & thermal diffusivity [m^2/s]
    return Cp * μ / k

def trans_Sc(μ, ρ, D):
    """
    Calculate the Schmidt number [-] from fluid viscosity [kg/(s m)], density [kg/m^3] & mass
    diffusivity [m^2/s].
    """
    # ν / D  # kinematic viscosity [m^2/s] & mass diffusivity [m^2/s]
    return μ / (ρ * D)

def turb_Nu_text(Re, Pr, ε, c=2.06, n=-0.575):
    """
    Calculate the turbulent Nusselt (or Sherwood) number [-] from the Reynolds & Prandtl (or
    Schmidt) numbers and void fraction (all [-]) for gas flow over a packed bed of spheres using
    the empirical correlation restated in Incropera & DeWitt (and other textbooks).
    Reported range of applicability: Pr approx. 0.7 & 90 <= Re <= 4000.
    Further suggested to multiply by 0.79 for cylinders of aspect ratio one, and 0.71 for cubes.
    """
    Nu = c * Re**(n + 1) * Pr**(1 / 3) / ε
    return Nu

def turb_Nu_finlayson(Re, Pr, ε, c0=0.36, c1=0.58, n1=0.48, c2=0.29, c3=0.028):
    """
    Calculate the turbulent Nusselt (or Sherwood) number [-] from the Reynolds & Prandtl (or
    Schmidt) numbers and void fraction (all [-]) for fixed spherical pellets using the
    empirical correlation proposed by Chang & Finlayson, 1987.
    Reported range of applicability: 7.3e-3 <= Pr <= 1.0e4 & 0.01 <= Re <= 50
    """
    m = c2 + c3 * log10(Re)
    Nu = (c0 + c1 * Re**n1) * Pr**m
    return Nu

def turb_Nu_Kramers(Re, Pr, ε, c0=1.26, c1=0.054, c2=0.8, n2=0.2):
    """
    Calculate the turbulent Nusselt (or Sherwood) number [-] from the Reynolds & Prandtl (or
    Schmidt) numbers and void fraction (all [-]) for fixed spherical pellets using the
    empirical correlation proposed by Thoenes and Kramers, 1958.
    Reported range of applicability: 100 <= Re <= 3500.
    """
    Nu = c0 * Re**(1/3) * Pr**(1/3) + c1 * Re**0.8 * Pr**0.4 + c2 * Re**n2
    return Nu

# Model for the mass-transfer flux through the film (at the catalyst surface):
def film_flux_x(X, u, rxn_rate_law, pore_eff, turb_Nu, θ, Ctot, A2V, Dse, guess):
    """
    Solve for the species mole fractions at the surface of the catalyst —
    with the film mass-transfer coupled to the pore reaction/diffusion.
    """
    # Explicitly unpack the more important parameters from θ & calculate a few parameters:
    P, T, R = θ.thermo  # gas pressure [Pa], temperature [K] and the gas constant [J / (mol K)]
    n_spec = spec_molar_masses.n_spec  # number of chemical species [-]
    M = spec_molar_masses(θ.M_atom)  # species molar masses [kg / mol]
    ρ = mix_molar_mass(X, θ.M_atom) * Ctot  # gas mass density [kg / m^3]
    μ_visc = θ.μ_visc  # gas viscosity [Pa s]
    n_rxn = rxn_rate_law.n_rxn  # number of chemical reactions [-]
    ν_rxn = rxn_rate_law.ν_rxn  # chemical reaction stoichiometry [-]
    Ap2Af = θ.ρ_cat * θ.a_cat / A2V  # pore total surface area over film area [-]
    rxn_rate_params = set_default_params(rxn_rate_law, θ)  # dict of reaction rate params.

    # Mass transfer through film & pore reaction/diffusion (as a residual):
    def rxn_diff(unknowns, xδ, u, residual=True):
        """Calculate a residual for the fluxes across the film."""
        # Unpack the unknown species mole fractions (all but one):
        x0 = empty(n_spec)
        x0[:-1] = unknowns
        x0[-1] = 1 - x0[:-1].sum()  # based on the constraint that x0.sum() = 1 (mole fractions)
        # print(f'{x0 = }')

        # Setup:
        xbar = (x0 + xδ) / 2  # mid-value mole fractions [-]
        Re = trans_Re(ρ, u, Dse, μ_visc)
        Sc = trans_Sc(μ_visc, ρ, θ.D)
        Sh = turb_Nu(Re, Sc, θ.ε_void, *θ.turb_bed)
        k = (Sh * θ.D) / Dse
        k /= 100  # WARNING: This artificially amplifies the film mass-transfer resistance!

        J = Ctot * k * (x0 - xδ)  # Newton's law for mass tranfer
        Ntot = -(M * J).sum() / mix_molar_mass(xbar, θ.M_atom)  # M @ N = 0 (consevation of mass)
        N = J + xbar * Ntot

        rxn_rate = rxn_rate_law(x0, P, T, **rxn_rate_params)
        ε_pore = pore_eff(θ.D_pore, θ.L_pore, θ.D, x0, P, T, Ctot, rxn_rate_law, **rxn_rate_params)
        # ε_pore = ones(n_rxn)  # FOR DEBUGGING ONLY!
        rate_eff = ε_pore * rxn_rate

        if residual:
            resid = empty(n_spec)
            resid = N[:-1] - Ap2Af * (ν_rxn @ rate_eff)[:-1]  # equate mass transfer & rxn rates
            resid *= 1e4  # scale the molar flux residuals to ~unity
            return resid
        else:
            return N, x0, rate_eff, ε_pore
    # return rxn_diff(guess, X, u, False)  # FOR TESTING!

    # res = fsolve(rxn_diff, guess, args=(X, T, u))
    res = least_squares(rxn_diff, guess, bounds=(0, +inf), xtol=1e-4, max_nfev=5000, args=(X, u))
    if not res.success:
        print(res)

    N, x0, rxn_rate, ε_pore = rxn_diff(res.x, X, u, False)
    # print('N & alt:', N, Ap2Af * (ν_rxn @ rxn_rate))
    # print('q & alt:', q, Ap2Af * (-θ.ΔH_rxn @ rxn_rate))
    rxn_rateδ = rxn_rate_law(X, P, T, **rxn_rate_params)
    ε_film = rxn_rate / (ε_pore * rxn_rateδ)
    return N, x0, rxn_rate, ε_film, ε_pore

# Model for the transport along the length of the pipe:
def rxr_plugflow_x(x_in, u_in, thermo_c, rxn_rate_law, pore_eff, cat_geom, turb_Nu, θ, n_z):
    """
    Model the chemical reactions as the species flow down a packed-bed tubular reactor.
    Use the plug-flow assumption.
    """
    # Explicitly unpack the more important parameters from θ & calculate a few parameters:
    P, T, R = θ.thermo  # gas pressure [Pa], temperature [K] and the gas constant [J / (mol K)]
    n_spec = spec_molar_masses.n_spec  # number of chemical species [-]
    n_rxn = rxn_rate_law.n_rxn  # number of chemical reactions [-]
    ν_rxn = rxn_rate_law.ν_rxn  # chemical reaction stoichiometry [-]

    # Global models
    Ctot = thermo_c(*θ.thermo)  # Total molar concentration from EOS
    A2V, Dse = cat_geom(*θ.cat_geom)  # Pellet outer area to vol. ratio & sphere-equivalent diam.

    # Plug-flow model (initial-value problem from the inlet of the reactor to the outlet):
    Zrxr, Δz = linspace(0, θ.l_rxr, n_z, retstep=True)
    Nrxr = empty((n_z, n_spec))
    Xrxr = empty((n_z, n_spec))
    Ncat = empty((n_z, n_spec))
    Xcat = empty((n_z, n_spec))
    ε_film = empty((n_z, n_rxn))
    ε_pore = empty((n_z, n_rxn))
    Nrxr[0] = u_in * Ctot * x_in
    Xrxr[0] = x_in
    u = u_in
    guess = array([0.04, 0.05, 0.01, 0.01, 0.01])  # initial guess — surface mole fractions [-]
    Ni = empty(n_spec)
    for i in range(1, n_z):
        guess = x_in[:-1].copy()  # initial guess — surface mole fractions [-]
        res = film_flux_x(Xrxr[i-1], u, rxn_rate_law, pore_eff, turb_Nu, θ, Ctot, A2V, Dse, guess)
        Ncat[i - 1], Xcat[i-1], rxn_rate, ε_film[i - 1], ε_pore[i - 1] = res
        Nrxr[i] = Nrxr[i-1] + Δz * (1 - θ.ε_void) * A2V * Ncat[i - 1]
        Xrxr[i] = Nrxr[i] / Nrxr[i].sum()
        guess = Xcat[i - 1, :-1].copy()
    return Zrxr, Nrxr, Xrxr, Ncat, Xcat, ε_film, ε_pore


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from parameter_set import ParameterSet, get_default_params, set_default_params

    # Submodel selections:
    thermo_c = thermo_c_ideal
    rxn_rate = rxn_rate_Klugherz
    pore_eff = pore_eff_thiele1
    cat_geom = cat_geom_sphere
    turb_Nu = turb_Nu_text

    # Parameter values:
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

    # Operational calculations:
    Ctot = thermo_c(*θ.thermo)
    n_dot = array([80.5, 91.5, 0, 0, 0, 1274]) / 30  # molar feed rates — indust. reactor [kmol/hr]
    n_dot *= 1e3 / (60 * 60)  # convert to [mol/s]
    x_in = n_dot / n_dot.sum()  # reactor-inlet mole fractions [-]
    Q_in = n_dot.sum() / Ctot  # volumetric feed rates — industrial reactor [m^3/s]
    A_tube = π * θ.r_rxr**2  # single-reactor-tube cross-sectional area [m^2]
    n_tubes = 2781  # number of tubes — industrial reactor [-]
    u_in = Q_in / (n_tubes * A_tube)  # superficial velocity [m/s]
    print(f'inlet superficial velocity = {u_in:.2f} m/s')

    # Run the coupled model:
    n_z = 600
    out = rxr_plugflow_x(x_in, u_in, thermo_c, rxn_rate, pore_eff, cat_geom, turb_Nu, θ, n_z)
    Zrxr, Nrxr, Xrxr, Ncat, Xcat, ε_film, ε_pore = out
    
    # Plot the results:
    plt.figure()
    plt.plot(Zrxr[:-1], Ncat[:-1], linestyle='-', linewidth=2.0)
    plt.gca().set_prop_cycle(None)
    # plt.plot(Zrxr[1:], Xcat[1:, :-1], linestyle='--', linewidth=2.0)
    plt.grid(True, alpha=0.25)
    plt.xlim(Zrxr[0], Zrxr[-1])
    # plt.ylim(0, None)
    plt.xlabel('distance into the reactor [m]', fontsize=16)
    plt.ylabel('rxn. molar flux [mol. / (s m$^2$)]', fontsize=16)
    plt.legend(spec_names, fontsize=12)

    plt.figure()
    plt.plot(Zrxr, Xrxr[:, :-1], linestyle='-', linewidth=2.0)
    plt.gca().set_prop_cycle(None)
    plt.plot(Zrxr[:-1], Xcat[:-1, :-1], linestyle='--', linewidth=2.0)
    plt.grid(True, alpha=0.25)
    plt.xlim(Zrxr[0], Zrxr[-1])
    plt.ylim(0, None)
    plt.xlabel('distance into the reactor [m]', fontsize=16)
    plt.ylabel('mole fraction [-]', fontsize=16)
    plt.legend(spec_names[:-1], fontsize=12)
    # plt.savefig('profile_reacting.pdf', transparent=True)

    plt.figure()
    plt.plot(Zrxr, Xrxr[:, -1], linestyle='-', linewidth=2.0, label='N$_2$ - bulk')
    plt.gca().set_prop_cycle(None)
    plt.plot(Zrxr[:-1], Xcat[:-1, -1], linestyle='--', linewidth=2.0, label='N$_2$ - cat.')
    plt.grid(True, alpha=0.25)
    plt.xlim(Zrxr[0], Zrxr[-1])
    # plt.ylim(0, None)
    plt.xlabel('distance into the reactor [m]', fontsize=16)
    plt.ylabel('mole fraction [-]', fontsize=16)
    plt.legend(fontsize=12)
    # plt.savefig('profile_N2.pdf', transparent=True)

    plt.figure()
    plt.plot(Zrxr[:-1], ε_pore[:-1], linestyle='-', linewidth=2.0)
    plt.gca().set_prop_cycle(None)
    plt.plot(Zrxr[:-1], ε_film[:-1], linestyle='--', linewidth=2.0)
    plt.grid(True, alpha=0.25)
    plt.xlim(Zrxr[0], Zrxr[-1])
    plt.ylim(0, 2)
    plt.xlabel('distance into the reactor [m]', fontsize=16)
    plt.ylabel('effectiveness factor [-]', fontsize=16)
    plt.legend([f'rxn. #{i + 1}' for i in range(rxn_rate.n_rxn)], fontsize=12)
    
    plt.show()