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

from numpy import array, empty, linspace, where, add, sqrt, exp, inf, pi as π
from numpy.linalg import solve
from scipy.optimize import least_squares  #, fsolve

from parameter_set import ParameterSet, get_default_params, set_default_params
from reactor_modeling import (thermo_c_ideal, spec_names, spec_molar_masses, mix_molar_mass,
                              calc_spec_rates, pore_eff_thiele1, trans_Re, trans_Pr, trans_Sc)

"""
Differences amongst the various models
----------------------------------------------------------------------------
|    sub-model    |    low     |          med.         |        high       |
| ---------------:|:----------:|:---------------------:|:-----------------:|
| **thermo.**     | ideal gas  | ideal gas             | ideal gas         |
| **cat. geom.**  | sphere     | cylinder              | annular cylinder  |
| **rxn. rate**   | Klugherz   | Klugherz w/ Arrhenius | combination       |
| **diffusion**   | const. $D$ | const. $D$            | Stefan-Maxwell    |
| **pore rxn.**   | Thiele 1st | Thiele 1st            | Thiele Linearized |
| **Nu pellets**  | *textbook* | ?                     | Gnielniski        |
| **Nu wall**     | Peters     | Peters                | Li                |
| **Temperature** | Const.     | heat transfer         | catalyst separate |
| **Advection**   | plug flow  | plug flow             | plug flow         |
----------------------------------------------------------------------------

I/U Map for the high-fidelity model
# I/U Map

| Sub-model/Expt.      | Parameter Name                                      | Symbol                             | Units              | Nominal Value(s)                                 | Uncertainty             | Expected Impact |
| -------------------- | :-------------------------------------------------: | :--------------------------------: | :----------------: | :----------------------------------------------: | :---------------------: | :-------------: |
| Thermodynamics       | gas constant                                        | $R$                                | m$^3$ Pa / (K mol) | $8.31446$                                        | $2 \times 10^{-6}$      | None to Small   |
|                      | pressure *(also an operating condition)*            | $P_{\mathrm{rxr}}$                 | Pa                 | $[1.0-3.0] \times 10^5$                          | $3\%$                   | Small           |
|                      | temperature *(in, also an operating condition)*     | $T_{\mathrm{rxr}}$                 | K                  | $[453-503]$                                      | $3^{\circ}$             | Small           |
| Species              | elemental molar masses (4)                          | $M_i$                              | g / mol            | [$12.01$, $1.01$, $16.00$, $14.01$]              | $0.1\%$                 | None to Small   |
|    & chem. thermo.   | species heat capacities (6)                         | $C_{P, i}$                         | J /(kg K)          | [$2160$, $970$, $1630$, $1020$, $1950$, $1050$]  | $15\%$                  | Small           |
| Transport            | viscosity, *molecular*                              | $\mu$                              | kg / (s m)         | $2.6 \times 10^{-5}$                             | $5\%$                   | Small           |
|                      | thermal conductivity, *molecular*                   | $k$                                | W / (m K)          | $0.041$                                          | $30\%$                  | Small           |
|                      | Fuller diffusion volumes, *molecular* (6)           | $V_{\mathrm{mol.}}$                | Å$^3$?             | [$41.0$, $16.3$, $47.2$, $26.7$, $13.1$, $18.5$] | $25\%$                  | Medium          |
|                      | Fuller constant                                     | $\alpha$                           | ?                  | $1.013 \times 10^{-2}$                           | $3\%$                   | Small           |
| Rxn. kinetics        | rate coefficients (2-3 depending)                   | $k_{\alpha}$                       | mol / (s m$^2$)    | *see individual model*                           | *should be positive*    | **Large**       |
|    & thermochem.     | pre-exponential & activation temp. (2 each dep.)    | $A_{\alpha}$, $E_{a, \alpha}$      | mol / (s m$^2$), K | *see individual model*                           | *should be positive*    | **Large**       |
|                      | rxn. order (per reactant, per rxn.)                 | $n_{\alpha}$                       | -                  | *see individual model*                           | *positive, rarely $>5$* | **Large**       |
|                      | adsorption coefficients (2-6 depending)             | $K_{i, \alpha}$                    | 1 / bar            | *see individual model*                           | *should be positive*    | **Large**       |
|                      | heats of reaction (2-3 depending)                   | $\Delta H^{\mathrm{rxn}}_{\alpha}$ | J / mol            | *see individual model*                           | $3\%$                   | Small           |
| Turbulence           | bed Nusselt number correlation params. (2-4)        | $C$, $m$ (dep.)                    | —                  | $c = 2.06$, $n = 0.575$                          | $20\%$                  | Small to Medium |
|                      | wall Nusselt number correlation params. (2-3)       | $C$, $m$ (dep.)                    | —                  | $c = 0.683$, $m = 0.466$                         | $20\%$                  | Small to Medium |
| Surface expts.       | saturation pressure of experimental adsorbate       | $P_{\mathrm{sat.}}$                | bar                |                                                  |                         |                 |
|                      | experimental pressures (2-5)                        | $P_{\mathrm{expt.}}$               | bar                |                                                  |                         |                 |
|                      | experimental temperature                            | $T_{\mathrm{expt.}}$               | K                  |                                                  |                         |                 |
|                      | experimental number adsorbed (2-5)                  | $n_{\mathrm{ad.}}$                 | mol                |                                                  |                         |                 |
| Catalyst             | mass density                                        | $\rho_{\mathrm{cat.}}$             | kg / m$^3$         | $285$                                            | $1\%$                   | Small to Medium |
|                      | specific surface area (w/ pores)                    | $a^{\mathrm{cat.}}$                | m$^2$ / kg         | $0.8 \times 10^3$                                | $10\%$                  | Medium          |
|                      | pellet height (macro)                               | $H_{\mathrm{pellet}}$              | m                  | $1.0 \times 10^{-2}$                             | $15\%$                  | Small           |
|                      | pellet outer diameter (macro)                       | $D_{o, \mathrm{pellet}}$           | m                  | $1.0 \times 10^{-2}$                             | $15\%$                  | Small           |
|                      | pellet inner diameter (macro)                       | $D_{i, \mathrm{pellet}}$           | m                  | $0.5 \times 10^{-2}$                             | $20\%$                  | Small           |
|                      | packed-catalyst void fraction in reactor            | $\varepsilon_{\mathrm{void}}$      | —                  | $0.75$                                           | $25\%$                  | Small to Medium |
|                      | pore diameter                                       | $D_{\mathrm{pore}}$                | m                  | $2 \times 10^{-6}$                               | $25\%$                  | Medium          |
|                      | pore length                                         | $L_{\mathrm{pore}}$                | m                  | $1 \times 10^{-3}$                               | $25\%$                  | Medium          |
| Reactor              | length                                              | $L_{\mathrm{rxr}}$                 | m                  | pilot rxr: $3.0$,  industrial rx: $12.8$         | $1\%$                   | Small           |
|                      | radius                                              | $r_{\mathrm{rxr}}$                 | m                  | $1.96 \times 10^{-2}$                            | $1\%$                   | Small           |
| Operating conditions | molar feed ($\ce{C_2H_4}$, $\ce{O_2}$ & $\ce{N_2}$) | $\dot{n}_{i, \mathrm{in}}$         | kmol / hr          | $80.5$, $91.5$, $1274$                           | $10\%$                  | Small to Medium |
|                      | inlet superficial velocity                          | $u_{\mathrm{in}}$                  | m / s              | $[0.5-5.0]$                                      | $10\%$                  | Small to Medium |
|                      | cooling temperature                                 | $T_{\mathrm{cool}}$                | K                  | $200-220$                                        | $8^{\circ}$             | Medium          |
|                      | ratio, outer-to-inner pipe-wall conv. coeffs.       | $T_{\mathrm{cool}}$                | -                  | $0.1 - 2$                                        | *unknown*               | Small to Medium |
"""

# Chemical reaction-rate models (rate-laws):
def rxn_rate_combo(x, P, T, a_Ag=1, Ke=2.305, Ko=6.506, Cprod=0.0, n1e=0.749, n1o=2.770, n2e=0.725,
                   n2o=2.895, Aeo=67.66, Ac=3.056e5, Ta_eo=8058, Ta_c=12_220, R31=0.784e-3,
                   ΔH_rxn=(-1e3 * array([106.7, 1323, 1323-106.7]))):
    """
    Calculate the reaction rates [mol-rxn. / (s m^2-cat.)] from species mole fraction [-],
    pressure [Pa] and temperature [K] using an alternative rate law (all sites available for
    adsorption, atomic-oxygen adsorption, Eley-Rideal for eth. ox., simple-Langmuir for CO2
    variable-order rxns., Arrhenius coefficients, support-based secondary oxidation).
    Parameter notes:
    Aeo — pre-exponential in eo rxn. [?]
    Ac — pre-exponential in c rxn. [?]
    Ta_eo — activation temperature [K]
    Ta_c — activation temperature [K]
    R31 — ratio of rate coefficients [kg bar^{1/2} / m^2]
          recommended values for different catalyst supports: 0.78e-3 for Al2O3 & 1.38e-3 for SiO2,
    ΔH_rxn — heats of reaction [J / mol-rxn.]  (Careful: use caution with mutable default args.).
    """
    # Convert from pressure [Pa] & mole fractions [-] to partial pressures [bar].
    Pe = x[0] * P * 1e-5
    Po = x[1] * P * 1e-5
    Peo = x[2] * P * 1e-5
    F = 1 + Ke * Pe + sqrt(Ko * Po) + Cprod + 1e-12
    θe = Pe
    θo = sqrt(Ko * Po) / F
    r1 = Aeo * exp(-Ta_eo / T) * θe**n1e * θo**n1o
    θe = Ke * Pe / F
    θo = sqrt(Ko * Po) / F
    r2 = Ac * exp(-Ta_c / T) * θe**n2e * θo**n2o
    r3 = a_Ag * R31 * Peo * r1 / (Pe * sqrt(Po))
    rates = array([r1, r2, r3])
    rxn_rate_combo.ΔH_rxn = ΔH_rxn  # provide access to these uncertain parameters
    return rates
# Certain parameters associated with the rxn. rate law needed elsewhere:
rxn_rate_combo.n_rxn=3  # number of chemical reactions
# reaction stoichiometry:
rxn_rate_combo.ν_rxn=array([[-1.0,-1.0, 0.0],  # ethylene
                            [-0.5,-3.0,-2.5],  # oxygen
                            [ 1.0, 0.0,-1.0],  # ethylene oxide
                            [ 0.0, 2.0, 2.0],  # carbon dioxide
                            [ 0.0, 2.0, 2.0],  # water
                            [ 0.0, 0.0, 0.0]]) # nitrogen

# Constitutive models for the diffusion coefficients:
def diff_bin_fuller(P, T, M, α_fuller=1.013e-2,
                    V_mol=array([41.04, 16.3, 47.15, 26.7, 13.1, 18.5])):
    """
    Calculate the species binary diffusion coefficients [m^2 / s] from preussure [Pa],
    Temperature [K] and molar masses [kg / mol] using the Fuller correlation.
    Parameter notes:
    α_fuller — constant in the Fuller equation
    V_mol — diffusion volumes [cm^3 / g-mol].
    """
    Dij = (α_fuller * T**1.75 * sqrt(add.outer(1 / (1e3 * M), 1 / (1e3 * M))) /
           (P * add.outer(V_mol**(1 / 3), V_mol**(1 / 3))**2))  # Fuller correlation
    return Dij

def diff_mix(Dij, x):
    """Calculate the species mixture-averaged diffusion coefficients [m^2 / s]."""
    Dinv = 1 / Dij
    Dinv.reshape(-1)[::(x.shape[0] + 1)] = 0
    Dmix = (1 - x) / (x * Dinv).sum(axis=1)  # Wilke model (ignores pore diffusion)
    return Dmix

def diff_pore_knudsen(T, D_pore, R, M):
    """
    Calculate the Knudsen pore diffusion coefficients [m^2 / s] from pore diameter [m],
    temperature [K], molar masses [kg / mol] and the gas constant [kg m^2 / (s^2 mol K)].
    """
    Dk = (D_pore / 3) * sqrt((8 * R * T) / (π * M))
    return Dk

def diff_mix_wpores(Dij, Dk, x):
    """
    Calculate the species mixture-averaged diffusion coefficients [m^2 / s] accounting for
    pore diffusion.
    """
    Dinv = 1 / Dij
    Dinv.reshape(-1)[::(x.shape[0] + 1)] = 0
    Dmix = 1 / ((x * Dinv).sum(axis=1) / (1 - x) + 1 / Dk)
    return Dmix

# def visc_Sutherland(T, T0=473.15, μ0=2.6e-5, Cμ=120):
#     """
#     Calculate viscosity 2.6e-5 [Pa s = kg / (s m)] from temp. [K] using Sutherland's formula.
#     """
#     return μ0 * ((T0 + Cμ) / (T + Cμ)) * (T / T0)**(3 / 2)

# Models for reaction/diffusion in pores (Thiele effect):
def pore_eff_thiele0(D_pore, L_pore, Dmix, x, P, T, Ctot, rxn_rate_law, **rate_params):
    """
    Calculate the effectiveness [-] of the pore surface area for reaction by coupling the surface
    rxn. with the diffusion down the length of the pore. Approximate each reaction by a piecewise-
    constant function — use the value of the reaction rate at the pore mouth and switch to zero
    reaction when any one reactant is depleated.
    """
    n_rxn = rxn_rate_law.n_rxn
    rate, _ = calc_spec_rates(x, P, T, rxn_rate_law, **rate_params)
    M = L_pore**2 * rate / (D_pore / 2 * Ctot * Dmix)
    z_crit_all = where(-M > x, L_pore * (1 - sqrt((x + M) / M)), L_pore)
    z_crit = z_crit_all[:n_rxn].min()  # species & rxns. happen to be ordered by primary reactant
    ε_pore = array([z_crit / L_pore] * n_rxn)
    return ε_pore

# def pore_eff_thieleL(D_pore, L_pore, Dmix, x, P, T, Ctot, rxn_rate_law, **rate_params):
#     """
#     Calculate the effectiveness [-] of the pore surface area for reaction by coupling the surface
#     rxn. with the diffusion down the length of the pore. Linearize each reaction by each of its
#     reactants (coefficient matrix) in order to determine the Theile modulus for coupled first-order
#     reactions.
#     """
#     ν_rxn = rxn_rate_law.ν_rxn
#     n_spec, n_rxn = ν_rxn.shape
#     K = empty((n_rxn, n_spec))
#     for i in range(n_rxn):
#         x_mod = where(ν_rxn[:, i] < 0, maximum(x, 1e-6), x)
#         ri = rxn_rate_law(x_mod, P, T, **rate_params)[i]
#         K[i] = where(ν_rxn[:, i] < 0, ri / x_mod, 0) / (ν_rxn[:, i] < 0).sum()  # only for reactants
#     A2 = -4 / (D_pore * Dmix) * (ν_rxn @ K)
#     Λ2, V = eig(A2)
#     Λ = sqrt(Λ2)
#     ε_Λ = where(Λ > 0, tanh(Λ * L_pore) / (Λ * L_pore), 0)
#     r_bar = Ctot * K @ (V * ε_Λ) @ inv(V) @ x
#     ε_pore = rxn_rate_law(x, P, T, *rate_params) / r_bar
#     return ε_pore

# def pore_eff_solve(D_pore, L_pore, Dij, x0, P, T, Ctot, rxn_rate_law, rate_params):
#     pass


# Models for the geometry of the catalyst pellets:
def cat_geom_annularcylinder(H, Do, Di):
    """
    Calculate the surface-area-to-volume ratio [1 / m] and sphere-equivalent diameter (diameter
    for a sphere of equal surface area) [m] for a right annular cylinder given its inner & outer
    diameters as well as its height.
    """
    A2V = 2 * (2 / (Do - Di) + 1 / H)
    Dse = sqrt((Do + Di) * ((Do - Di) / 2 + H))
    return A2V, Dse

# Transport & Turbulence models for the Sherwood & Nusselt numbers:
def turb_Nu_Gnielinski(Re, Pr, ε, fa=1.6, c0=0.664, c1=0.037, c2=2.443, n2=-0.1):
    """
    Calculate the turbulent Nusselt (or Sherwood) number [-] from the Reynolds & Prandtl (or
    Schmidt) numbers and void fraction (all [-]) for a packed bed of various shaped pellets
    (spheres, cylinders, Rachig rings, etc.) using the empirical correlation proposed by
    Gnielinski, 1982, and summarized by Gnielinski in The VDI Heat Atlas, 2nd edition, Springer-
    Verlag, 2010, p. 743 — https://link.springer.com/content/pdf/10.1007/978-3-540-77877-6.pdf.
    
    The velocity used in the Reynolds number, here, is the superficial velocity (so that once
    divided by void fraction, it becomes the interstitial velocity).
    The effective diameter used in Re & Nu (or Sh) is the diameter of the sphere with equivalent
    area.
    The factor fa represents a correction going from a single fixed particle to a packed bed —
    with careful consideration of wall channeling. Recommended values for fa:
    - mono-dispersed spheres, fa = 1 + 1.5 * (1 - ε);
    - cylinders with 0.24 < l/d < 1.2 and cubes, fa = 1.6;
    - Raschig rings, fa = 2.1;
    - Berl saddles, fa = 2.3.
    Reported range of applicability: 100 <= Re <= 1e4.
    """
    Nu_lam = c0 * (Re / ε)**(1 / 2) * Pr**(1 / 3)
    Nu_turb = c1 * (Re / ε)**0.8 * Pr / (1 + c2 * (Re / ε)**n2 * (Pr**(2 / 3) - 1))
    Nu = fa * (2 + sqrt(Nu_lam**2 + Nu_turb**2))
    return Nu

def turb_Nu_Li(Re, Pr, Dpipe2Dpellet, c0=0.16, n0=0.93):
    """
    Calculate the turbulent Nusselt number [-] from the Reynolds & Prandtl numbers and the
    diameter ratio (all [-]) for the inner pipe wall with a fixed bed of spherical or cylindrical
    pellets using the empirical correlation proposed by Li & Finlayson, 1977.
    Reported range of applicability:  20 <= Re <= 7600  and  3.3 <= (D_pipe/D_pellet) <=20.
    Reported empirical parameter values: c0=0.17, n0=0.79 for spheres and
                                         c0=0.16, n0=0.93 for cylinders (all [-]).
    """
    Nu = c0 * Re**n0
    return Nu

def turb_Nu_Peters(Re, Pr, Dpipe2Dpellet, c0=3.8, m0=0.39, n0=0.50):
    """
    Calculate the turbulent Nusselt number [-] from the Reynolds & Prandtl numbers and the
    diameter ratio (all [-]) for the inner pipe wall with a fixed bed of spherical or cylindrical
    pellets using the empirical correlation proposed by Peters, 1988.
    Reported range of applicability:  800 <= Re <= 8000  and  3 <= (D_pipe/D_pellet) <=11.
    Reported empirical parameter values: c0=4.9, m0=0.26, n0=0.45 for spheres and
                                         c0=3.8, m0=0.39, n0=0.50 for cylinders (all [-]).
    """
    Nu = c0 * Dpipe2Dpellet**m0 * Re**n0 * Pr**(1 / 3)
    return Nu

# Model for the mass-transfer flux through the film (at the catalyst surface):
def film_flux_xT(X, T, u, rxn_rate_law, diff_bin_coeffs, pore_eff, turb_Nu,
                 θ, Ctot, A2V, Dse, guess):
    """
    Solve for the species mole fractions & temperature at the surface of the catalyst —
    with the film mass/heat-transfer coupled to the pore reaction/diffusion.
    """
    # Explicitly unpack the more important parameters from θ & calculate a few parameters:
    P, R = θ.thermo  # gas pressure [Pa] and the gas constant [J / (mol K)]
    n_spec = spec_molar_masses.n_spec  # number of chemical species [-]
    M = spec_molar_masses(θ.M_atom)  # species molar masses [g / mol]
    ρ = mix_molar_mass(X, θ.M_atom) * Ctot  # gas mass density [kg / m^3]
    Cp = θ.heat_cap  # species heat capacities [J / (kg K)]
    k_cond = θ.k_conduct  # gas thermal conductivity [W / (m K)]
    μ_visc = θ.μ_visc  # gas viscosity [Pa s]
    n_rxn = rxn_rate_law.n_rxn  # number of chemical reactions [-]
    ν_rxn = rxn_rate_law.ν_rxn  # chemical reaction stoichiometry [-]
    Ap2Af = θ.ρ_cat * θ.a_cat / A2V  # pore total surface area over film area [-]
    rxn_rate_params = set_default_params(rxn_rate_law, θ)  # dict of reaction rate params.
    # Species binary diffusion coefficients:
    Dij = diff_bin_coeffs(P, T, M, θ.α_fuller, θ.V_mol)  # don't recalculate w/ small changes in T

    # Mass transfer through film & pore reaction/diffusion (as a residual):
    def rxn_diff(unknowns, xδ, Tδ, u, residual=True):
        """Calculate a residual for the fluxes across the film."""
        
        # Explicitly unpack the more important parameters from θ:
        # Unpack the unknown species mole fractions (all but one) and temperature:
        x0 = empty(n_spec)
        x0[:-1] = unknowns[:-1]
        x0[-1] = 1 - x0[:-1].sum()  # based on the constraint that x0.sum() = 1 (mole fractions)
        T0 = unknowns[-1]

        # Setup:
        # TODO: recalculate Ctot, ρ & Dij at mid & 0?
        xbar = (x0 + xδ) / 2  # mid-value mole fractions [-]
        Re = trans_Re(ρ, u, Dse, μ_visc)
        Sc_ij = trans_Sc(μ_visc, ρ, Dij)
        Sh_ij = turb_Nu(Re, Sc_ij, θ.ε_void, *θ.turb_bed)
        k_ij = (Sh_ij * Dij) / Dse
        k_ij /= 100  # WARNING: This artificailly amplifies the film mass-trasfer resistance!
        Cp_bar = (xbar * Cp * M).sum() / (xbar * M).sum()
        Pr = trans_Pr(Cp_bar, μ_visc, k_cond)
        Nu = turb_Nu(Re, Pr, θ.ε_void, *θ.turb_bed)
        h = (Nu * k_cond / Dse)
        h /= 4 / 3  # WARNING: This artificailly amplifies the film heat-trasfer resistance!
        
        # Stefan-Maxwell multi-component diffusion with a mass-tranfer coefficinet (Newton's law):
        A = -(1 / k_ij) * xbar.reshape((n_spec, 1))
        A.reshape(-1)[::n_spec + 1] += (1 / k_ij) @ xbar
        b = Ctot * (x0 - xδ)
        A[-1] = M
        b[-1] = 0  # constraint: M @ N = 0 (consevation of mass)
        N = solve(A, b)

        q = h * (T0 - Tδ)  # Newton's law of cooling

        rxn_rate = rxn_rate_law(x0, P, T0, **rxn_rate_params)
        # rxn_rate = rxn_rate_law(x0, P, Tδ, **rxn_rate_params)  # FOR DEBUGGING ONLY!
        Dk = diff_pore_knudsen(T0, θ.D_pore, R, M)
        Dmix = diff_mix_wpores(Dij, Dk, x0)
        ε_pore = pore_eff(θ.D_pore, θ.L_pore, Dmix, x0, P, T0, Ctot,
                          rxn_rate_law, **rxn_rate_params)
        # ε_pore = ones(n_rxn)  # FOR DEBUGGING ONLY!
        rate_eff = ε_pore * rxn_rate

        if residual:
            resid = empty(n_spec)
            resid[:-1] = N[:-1] - Ap2Af * (ν_rxn @ rate_eff)[:-1]
            resid[:-1] *= 1e4  # scale the molar flux residuals to ~unity
            resid[-1] = q - Ap2Af * (-θ.ΔH_rxn @ rate_eff)
            # resid[-1] *= 1e-3  # scale the heat flux residuals to ~unity
            return resid
        else:
            return N, x0, q, T0, rate_eff, ε_pore
    # return rxn_diff(guess, X, T, u, False)  # FOR TESTING!
    
    # res = fsolve(rxn_diff, guess, args=(X, T, u, rxn_rate_law, pore_eff, turb_Nu, θ, Ctot, Dij))
    res = least_squares(rxn_diff, guess, bounds=(0, inf), xtol=1e-4, max_nfev=5000, args=(X, T, u))
    if not res.success:
        print(res)

    N, x0, q, T0, rxn_rate, ε_pore = rxn_diff(res.x, X, T, u, False)
    # print('N & alt:', N, Ap2Af * (ν_rxn @ rxn_rate))
    # print('q & alt:', q, Ap2Af * (-θ.ΔH_rxn @ rxn_rate))
    rxn_rateδ = rxn_rate_law(X, P, T0, **rxn_rate_params)
    ε_film = rxn_rate / (ε_pore * rxn_rateδ)

    return N, x0, q, T0, rxn_rate, ε_film, ε_pore

# Model for the transport along the length of the pipe:
def rxr_plugflow_xT(x_in, T_in, u_in, thermo_c, rxn_rate_law, diff_bin_coeffs, pore_eff,
                    cat_geom, turb_Nu_bed, turb_Nu_wall, θ, n_z):
    """
    Model the chemical reactions & thermal variation as the species flow down a packed-bed
    tubular reactor. Use the plug-flow assumption.
    """
    # Explicitly unpack the more important parameters from θ & calculate a few parameters:
    P, R = θ.thermo  # gas pressure [Pa] and the gas constant [J / (mol K)]
    n_spec = spec_molar_masses.n_spec  # number of chemical species [-]
    M = spec_molar_masses(θ.M_atom)  # species molar masses [g / mol]
    Cp = θ.heat_cap  # species heat capacities [J / (kg K)]
    k_cond = θ.k_conduct  # gas thermal conductivity [W / (m K)]
    μ_visc = θ.μ_visc  # gas viscosity [Pa s]
    n_rxn = rxn_rate_law.n_rxn  # number of chemical reactions [-]

    # Global models
    A2V, Dse = cat_geom(*θ.cat_geom)  # Pellet outer area to vol. ratio & sphere-equivalent diam.

    # Plug-flow model (initial-value problem from the inlet of the reactor to the outlet):
    Zrxr, Δz = linspace(0, θ.l_reactor, n_z, retstep=True)
    Nrxr = empty((n_z, n_spec))
    Xrxr = empty((n_z, n_spec))
    Trxr = empty(n_z)
    Ncat = empty((n_z, n_spec))
    Xcat = empty((n_z, n_spec))
    Tcat = empty(n_z)
    ε_film = empty((n_z, n_rxn))
    ε_pore = empty((n_z, n_rxn))
    Ctot = thermo_c(P, T_in, R)  # Total molar concentration from EOS
    Nrxr[0] = u_in * Ctot * x_in
    Xrxr[0] = x_in
    Trxr[0] = T_in
    guess = array([*x_in[:-1], T_in])  # initial guess for catalyst concentration & temperature
    Ni = empty(n_spec)
    for i in range(1, n_z):
        Ctot = thermo_c(P, Trxr[i-1], R)  # Total molar concentration from EOS
        Mbar = mix_molar_mass(Xrxr[i-1], θ.M_atom)
        ρ = Mbar * Ctot
        u = Nrxr[i-1].sum() / Ctot
        Re = trans_Re(ρ, u, Dse, μ_visc)
        Cp_bar = (Xrxr[i-1] * Cp * M).sum() / Mbar
        Pr = trans_Pr(Cp_bar, μ_visc, k_cond)
        Nu = turb_Nu_wall(Re, Pr, 2 * θ.r_reactor / Dse, *θ.turb_wall)
        h_pipe = Nu * k_cond / Dse
        res = film_flux_xT(Xrxr[i-1], Trxr[i-1], u, rxn_rate_law, diff_bin_coeffs, pore_eff,
                           turb_Nu_bed, θ, Ctot, A2V, Dse, guess)
        Ncat[i - 1], Xcat[i - 1], q_rxn, Tcat[i - 1], r_rxn, ε_film[i - 1], ε_pore[i - 1] = res
        Nrxr[i] = Nrxr[i-1] + Δz * (1 - θ.ε_void) * A2V * Ncat[i - 1]
        Xrxr[i] = Nrxr[i] / Nrxr[i].sum()
        Trxr[i] = Trxr[i-1] + Δz / (u * ρ * Cp_bar) * ((1 - θ.ε_void) * A2V * q_rxn
                  - (2 / θ.r_reactor) * (1 / (1 + 1/10)) * h_pipe * (Trxr[i-1] - θ.T_cool))
        guess[:-1] = Xcat[i - 1, :-1]
        guess[-1] = Tcat[i - 1]
    return Zrxr, Nrxr, Xrxr, Trxr, Ncat, Xcat, Tcat, ε_film, ε_pore


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Submodel selection:
    thermo_c = thermo_c_ideal
    rxn_rate = rxn_rate_combo
    diff_bin_coeffs = diff_bin_fuller
    pore_eff = pore_eff_thiele1
    cat_geom = cat_geom_annularcylinder
    turb_Nu_bed = turb_Nu_Gnielinski
    turb_Nu_wall = turb_Nu_Li

    # Parameter values:
    PSS= type('ParameterSetSubclass1', (ParameterSet,), dict())
    θ = PSS(# Thermodynamic properties:
            thermo=dict(P=(2 * 101.325e3),  # pressure [Pa]
                        R=get_default_params(thermo_c)['R']),  # gas constant
            # Chemical species & transport properties:
            spec=dict(**get_default_params(spec_molar_masses),
                      # species molar heat capacities [J / (kg K)] (C2H4, O2, C2H4O, CO2, H2O, N2)
                      heat_cap=array([2160, 970, 1630, 1020, 1950, 1050]),
                      diff_bin_coeffs=get_default_params(diff_bin_coeffs),
                      k_conduct=0.041,  # thermal conductivity of the gas [W / (m K)]
                      μ_visc=2.6e-5),  # viscosity [Pa s = kg / (s m)]
            # Chemical reaction properties & parameters:
            rxn=get_default_params(rxn_rate),
            # Turbulence parameters:
            turb_bed=get_default_params(turb_Nu_bed),
            turb_wall=get_default_params(turb_Nu_wall),
            # Catalyst properties:
            cat=dict(ρ_cat=881,  # catalyst density [kg / m^3]
                     a_cat=0.8e3,  # catalyst surface area (including pores) per mass [m^2/kg]
                     ε_void=0.75,  # as-packed void fraction (gaps-between pellets) [-]
                     cat_geom=dict(H_pellet = 2e-2,  # pellet height [m]
                                   Do_pellet = 2e-2,  # pellet outer diameter [m]
                                   Di_pellet = 1e-2),  # pellet inner diameter [m]
                     D_pore=4e-6,  # diameter of the pores [m]
                     # D_pore = 40e-9,  # diameter of the pores [m]  # WARNING: for testing only!
                     L_pore=1e-3),  # characteristic pore length [m],
            # Reactor properties & operation:
            rxr=dict(l_reactor=12.8,  # reactor length [m]
                    # l_reactor=2.0,  # reactor length [m]
                    r_reactor=1.956e-2,  # reactor radius [m]
                    T_cool=(200 + 273.15)))  # temperature of the cooling fluid for the reactor [K]

    T_in = 200 + 273.15  # temperature [K]
    Ctot = thermo_c(θ.P, T_in, θ.R)

    # Operational calculations:
    n_dot = array([80.5, 91.5, 0, 0, 0, 1274]) / 30  # molar feed rates — indust. reactor [kmol/hr]
    # n_dot = array([80.5, 91.5, 0, 0, 0, 1274]) / 5  # molar feed rates — indust. reactor [kmol/hr]
    n_dot *= 1e3 / (60 * 60)  # convert to [mol/s]
    x_in = n_dot / n_dot.sum()  # reactor-inlet mole fractions [-]
    Q_in = n_dot.sum() / Ctot  # volumetric feed rates — industrial reactor [m^3/s]
    A_tube = π * θ.r_reactor**2  # single-reactor-tube cross-sectional area [m^2]
    n_tubes = 2781  # number of tubes — industrial reactor [-]
    u_in = Q_in / (n_tubes * A_tube)  # superficial velocity [m/s]
    print(f'inlet superficial velocity = {u_in:.2f} m/s')

    # Run the coupled model:
    n_z = 600
    out = rxr_plugflow_xT(x_in, T_in, u_in, thermo_c, rxn_rate, diff_bin_coeffs, pore_eff,
                        cat_geom, turb_Nu_bed, turb_Nu_wall, θ, n_z)
    Zrxr, Nrxr, Xrxr, Trxr, Ncat, Xcat, Tcat, ε_film, ε_pore = out

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

    plt.figure()
    plt.plot([Zrxr[0], Zrxr[-1]], [θ.T_cool - 273.15, θ.T_cool - 273.15],
            linewidth=2.0, label='coolant')
    plt.plot(Zrxr, Trxr - 273.15, linewidth=2.0, label='gas')
    plt.plot(Zrxr[:-1], Tcat[:-1] - 273.15, linewidth=2.0, label='catalyst')
    plt.grid(True, alpha=0.25)
    plt.xlim(Zrxr[0], Zrxr[-1])
    # plt.ylim(0, None)
    plt.xlabel('distance into the reactor [m]', fontsize=16)
    plt.ylabel('Temperature [C]', fontsize=16)
    plt.legend(fontsize=12)

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