import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Tuple, Dict

# -------------------------------------------------------------
# Utility functions for normalization
# -------------------------------------------------------------

def normalize(value: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
    return (value - min_v) / (max_v - min_v)

def denormalize(norm_value: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
    return norm_value * (max_v - min_v) + min_v

# -------------------------------------------------------------
# Physics informed neural network for coupled thermal-fluid flow
# -------------------------------------------------------------
class PINN(nn.Module):
    """Basic PINN with 5 outputs: T, P, Ux, Uy, Uz."""

    def __init__(self, layers: Tuple[int, ...], initial_dsigma_dT: float,
                 params: Dict[str, float]):
        super().__init__()
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers_list.append(nn.SiLU())
        self.dsigma_dT = nn.Parameter(torch.tensor([initial_dsigma_dT], dtype=torch.get_default_dtype()))

        # store process parameters
        self.rho_val = params.get('rho', 7800.0)
        self.mu_visc_val = params.get('mu', 6e-3)
        self.P_laser_val = params.get('P_laser', 100.0)
        self.r_beam_val = params.get('r_beam', 1e-3)
        self.v_scan_val = params.get('v_scan', 1e-3)
        self.eta_abs_val = params.get('eta', 0.3)
        self.h_conv_val = params.get('h_conv', 20.0)
        self.eps_em_val = params.get('eps_em', 0.3)
        self.sigma_sb_val = params.get('sigma_sb', 5.67e-8)
        self.T_ambient_val = params.get('T_ambient', 300.0)
        self.T_solidus_val = params.get('T_solidus', 1600.0)
        self.T_liquidus_val = params.get('T_liquidus', 1700.0)
        # domain boundaries for laser position
        self.X_MIN = params.get('X_MIN', 0.0)
        self.X_MAX = params.get('X_MAX', 1.0)
        self.Y_MIN = params.get('Y_MIN', 0.0)
        self.Y_MAX = params.get('Y_MAX', 1.0)
        self.q_laser_char_peak_absorbed = 2 * self.eta_abs_val * self.P_laser_val / (torch.pi * self.r_beam_val ** 2)

    # ---------------------------------------------------------
    # Temperature dependent material properties
    # ---------------------------------------------------------
    def liquid_fraction(self, T_val: torch.Tensor) -> torch.Tensor:
        """Smoothed Heaviside function for melt fraction."""
        delta_T = self.T_liquidus_val - self.T_solidus_val
        return torch.clamp((T_val - self.T_solidus_val) / delta_T, 0.0, 1.0)

    def get_cp(self, T_val: torch.Tensor) -> torch.Tensor:
        Cp_solid = 0.2441 * T_val + 338.39
        Cp_liquid = torch.full_like(T_val, 709.25)
        fL = self.liquid_fraction(T_val)
        return (1 - fL) * Cp_solid + fL * Cp_liquid

    def get_k_therm(self, T_val: torch.Tensor) -> torch.Tensor:
        k_solid = 3.0e-5 * T_val ** 2 - 0.0366 * T_val + 18.588
        k_liquid = torch.full_like(T_val, 30.078)
        fL = self.liquid_fraction(T_val)
        return (1 - fL) * k_solid + fL * k_liquid

    # ---------------------------------------------------------
    # Neural network forward pass
    # ---------------------------------------------------------
    def forward(self, x_n: torch.Tensor, y_n: torch.Tensor, z_n: torch.Tensor, t_n: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        inp = torch.cat([x_n, y_n, z_n, t_n], dim=1)
        out = inp
        for layer in self.layers_list:
            out = layer(out)
        # split outputs
        Tn = torch.sigmoid(out[:, 0:1])
        Pn = torch.tanh(out[:, 1:2])
        Unx = torch.tanh(out[:, 2:3])
        Uny = torch.tanh(out[:, 3:4])
        Unz = torch.tanh(out[:, 4:5])
        return Tn, Pn, Unx, Uny, Unz

    # ---------------------------------------------------------
    # Laser heat source
    # ---------------------------------------------------------
    def laser_source(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_center = self.X_MIN + self.v_scan_val * t
        y_center = self.Y_MIN + (self.Y_MAX - self.Y_MIN) / 2.0
        r_sq = (x - x_center) ** 2 + (y - y_center) ** 2
        return self.q_laser_char_peak_absorbed * torch.exp(-2 * r_sq / self.r_beam_val ** 2)

    # ---------------------------------------------------------
    # PDE residuals for mass, momentum and energy
    # ---------------------------------------------------------
    def pde_losses_fluid(self, x_c_n: torch.Tensor, y_c_n: torch.Tensor, z_c_n: torch.Tensor, t_c_n: torch.Tensor,
                         norm_params: Dict[str, Dict[str, float]],
                         fluid_norm_params: Dict[str, Dict[str, float]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PDE residual losses."""
        # scaling helpers
        x_orig = denormalize(x_c_n, **norm_params['x'])
        y_orig = denormalize(y_c_n, **norm_params['y'])
        z_orig = denormalize(z_c_n, **norm_params['z'])
        t_orig = denormalize(t_c_n, **norm_params['t'])

        Tn, Pn, Unx, Uny, Unz = self.forward(x_c_n, y_c_n, z_c_n, t_c_n)

        T = denormalize(Tn, **norm_params['T'])
        P = denormalize(Pn, **fluid_norm_params['P'])
        Ux = denormalize(Unx, **fluid_norm_params['U'])
        Uy = denormalize(Uny, **fluid_norm_params['U'])
        Uz = denormalize(Unz, **fluid_norm_params['U'])

        cp_val = self.get_cp(T)
        k_val = self.get_k_therm(T)

        # derivative helper
        def safe_grad(outputs, inputs):
            grad_val = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                            create_graph=True, retain_graph=True, allow_unused=True)[0]
            if grad_val is None:
                grad_val = torch.zeros_like(inputs)
            return grad_val

        # first order derivatives in normalized space
        dTn_dx = safe_grad(Tn, x_c_n)
        dTn_dy = safe_grad(Tn, y_c_n)
        dTn_dz = safe_grad(Tn, z_c_n)
        dTn_dt = safe_grad(Tn, t_c_n)

        dUnx_dx = safe_grad(Unx, x_c_n)
        dUnx_dy = safe_grad(Unx, y_c_n)
        dUnx_dz = safe_grad(Unx, z_c_n)
        dUnx_dt = safe_grad(Unx, t_c_n)

        dUny_dx = safe_grad(Uny, x_c_n)
        dUny_dy = safe_grad(Uny, y_c_n)
        dUny_dz = safe_grad(Uny, z_c_n)
        dUny_dt = safe_grad(Uny, t_c_n)

        dUnz_dx = safe_grad(Unz, x_c_n)
        dUnz_dy = safe_grad(Unz, y_c_n)
        dUnz_dz = safe_grad(Unz, z_c_n)
        dUnz_dt = safe_grad(Unz, t_c_n)

        dPn_dx = safe_grad(Pn, x_c_n)
        dPn_dy = safe_grad(Pn, y_c_n)
        dPn_dz = safe_grad(Pn, z_c_n)

        # second order derivatives for temperature and velocities
        d2Tn_dx2 = safe_grad(dTn_dx, x_c_n)
        d2Tn_dy2 = safe_grad(dTn_dy, y_c_n)
        d2Tn_dz2 = safe_grad(dTn_dz, z_c_n)

        d2Unx_dx2 = safe_grad(dUnx_dx, x_c_n)
        d2Unx_dy2 = safe_grad(dUnx_dy, y_c_n)
        d2Unx_dz2 = safe_grad(dUnx_dz, z_c_n)

        d2Uny_dx2 = safe_grad(dUny_dx, x_c_n)
        d2Uny_dy2 = safe_grad(dUny_dy, y_c_n)
        d2Uny_dz2 = safe_grad(dUny_dz, z_c_n)

        d2Unz_dx2 = safe_grad(dUnz_dx, x_c_n)
        d2Unz_dy2 = safe_grad(dUnz_dy, y_c_n)
        d2Unz_dz2 = safe_grad(dUnz_dz, z_c_n)

        # apply chain rule scaling from normalized to physical
        x_range = norm_params['x']['max_v'] - norm_params['x']['min_v']
        y_range = norm_params['y']['max_v'] - norm_params['y']['min_v']
        z_range = norm_params['z']['max_v'] - norm_params['z']['min_v']
        t_range = norm_params['t']['max_v'] - norm_params['t']['min_v']

        T_range = norm_params['T']['max_v'] - norm_params['T']['min_v']
        U_range = fluid_norm_params['U']['max_v'] - fluid_norm_params['U']['min_v']
        P_range = fluid_norm_params['P']['max_v'] - fluid_norm_params['P']['min_v']

        dT_dx = dTn_dx * T_range / x_range
        dT_dy = dTn_dy * T_range / y_range
        dT_dz = dTn_dz * T_range / z_range
        dT_dt = dTn_dt * T_range / t_range

        dUx_dx = dUnx_dx * U_range / x_range
        dUx_dy = dUnx_dy * U_range / y_range
        dUx_dz = dUnx_dz * U_range / z_range
        dUx_dt = dUnx_dt * U_range / t_range

        dUy_dx = dUny_dx * U_range / x_range
        dUy_dy = dUny_dy * U_range / y_range
        dUy_dz = dUny_dz * U_range / z_range
        dUy_dt = dUny_dt * U_range / t_range

        dUz_dx = dUnz_dx * U_range / x_range
        dUz_dy = dUnz_dy * U_range / y_range
        dUz_dz = dUnz_dz * U_range / z_range
        dUz_dt = dUnz_dt * U_range / t_range

        dP_dx = dPn_dx * P_range / x_range
        dP_dy = dPn_dy * P_range / y_range
        dP_dz = dPn_dz * P_range / z_range

        d2T_dx2 = d2Tn_dx2 * T_range / (x_range ** 2)
        d2T_dy2 = d2Tn_dy2 * T_range / (y_range ** 2)
        d2T_dz2 = d2Tn_dz2 * T_range / (z_range ** 2)

        d2Ux_dx2 = d2Unx_dx2 * U_range / (x_range ** 2)
        d2Ux_dy2 = d2Unx_dy2 * U_range / (y_range ** 2)
        d2Ux_dz2 = d2Unx_dz2 * U_range / (z_range ** 2)

        d2Uy_dx2 = d2Uny_dx2 * U_range / (x_range ** 2)
        d2Uy_dy2 = d2Uny_dy2 * U_range / (y_range ** 2)
        d2Uy_dz2 = d2Uny_dz2 * U_range / (z_range ** 2)

        d2Uz_dx2 = d2Unz_dx2 * U_range / (x_range ** 2)
        d2Uz_dy2 = d2Unz_dy2 * U_range / (y_range ** 2)
        d2Uz_dz2 = d2Unz_dz2 * U_range / (z_range ** 2)

        # PDE residuals
        continuity_res = dUx_dx + dUy_dy + dUz_dz
        loss_mass = torch.mean(continuity_res ** 2)

        conv_Ux = Ux * dUx_dx + Uy * dUx_dy + Uz * dUx_dz
        lap_Ux = d2Ux_dx2 + d2Ux_dy2 + d2Ux_dz2
        mom_x_res = self.rho_val * (dUx_dt + conv_Ux) + dP_dx - self.mu_visc_val * lap_Ux

        conv_Uy = Ux * dUy_dx + Uy * dUy_dy + Uz * dUy_dz
        lap_Uy = d2Uy_dx2 + d2Uy_dy2 + d2Uy_dz2
        mom_y_res = self.rho_val * (dUy_dt + conv_Uy) + dP_dy - self.mu_visc_val * lap_Uy

        conv_Uz = Ux * dUz_dx + Uy * dUz_dy + Uz * dUz_dz
        lap_Uz = d2Uz_dx2 + d2Uz_dy2 + d2Uz_dz2
        mom_z_res = self.rho_val * (dUz_dt + conv_Uz) + dP_dz - self.mu_visc_val * lap_Uz

        loss_momentum = torch.mean(mom_x_res ** 2 + mom_y_res ** 2 + mom_z_res ** 2)

        convection_T = Ux * dT_dx + Uy * dT_dy + Uz * dT_dz
        lap_T = d2T_dx2 + d2T_dy2 + d2T_dz2
        energy_res = self.rho_val * cp_val * (dT_dt + convection_T) - k_val * lap_T
        loss_energy = torch.mean(energy_res ** 2)

        return loss_mass, loss_momentum, loss_energy
