"""JONSWAP-style wave induced velocity model for EasyUUV experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class JonswapWaveDisturbanceManager:
    """Lightweight JONSWAP/Airy velocity synthesizer.

    The manager samples a small deterministic set of wave components and returns
    horizontal induced velocity at the UUV position.  It is designed for
    reviewer-facing robustness experiments rather than high-cost CFD.
    """

    hs: float = 0.5
    fp: float = 0.1
    gamma: float = 3.3
    depth: float = 30.0
    direction: float = 0.0
    seed: int = 7
    num_frequencies: int = 12
    num_directions: int = 7
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.seed))

        f_min = max(0.03, 0.5 * float(self.fp))
        f_max = max(f_min + 1e-3, 3.0 * float(self.fp))
        frequencies = torch.linspace(f_min, f_max, self.num_frequencies, device=self.device)
        directions = torch.linspace(-math.pi / 2, math.pi / 2, self.num_directions, device=self.device) + float(
            self.direction
        )
        self.frequencies, self.directions = torch.meshgrid(frequencies, directions, indexing="ij")
        self.omega = 2.0 * math.pi * self.frequencies
        self.k = torch.clamp(self.omega**2 / 9.81, min=1e-6)
        self.kx = self.k * torch.cos(self.directions)
        self.ky = self.k * torch.sin(self.directions)

        spectrum = self._jonswap_spectrum(self.frequencies)
        directional = torch.clamp(torch.cos(self.directions - float(self.direction)), min=0.0) ** 2
        df = (f_max - f_min) / max(1, self.num_frequencies - 1)
        dtheta = math.pi / max(1, self.num_directions - 1)
        amplitude = torch.sqrt(torch.clamp(2.0 * spectrum * directional * df * dtheta, min=0.0))
        if self.hs > 0:
            amplitude = amplitude * float(self.hs)
        self.amplitude = amplitude
        self.phase = (2.0 * math.pi * torch.rand(self.frequencies.shape, generator=generator)).to(self.device)

    def _jonswap_spectrum(self, frequency: torch.Tensor) -> torch.Tensor:
        sigma = torch.where(frequency <= self.fp, torch.tensor(0.07, device=self.device), torch.tensor(0.09, device=self.device))
        alpha = 0.076 * max(float(self.hs), 1e-6) ** 2 * max(float(self.fp), 1e-6) ** 4
        exponent = -((frequency - float(self.fp)) ** 2) / (2.0 * sigma**2 * float(self.fp) ** 2 + 1e-12)
        frequency = torch.clamp(frequency, min=1e-4)
        return (
            alpha
            * 9.81**2
            / ((2.0 * math.pi) ** 4 * frequency**5)
            * torch.exp(-1.25 * (float(self.fp) / frequency) ** 4)
            * float(self.gamma) ** torch.exp(exponent)
        )

    def get_wave_velocity(self, position_w: torch.Tensor, time_s: torch.Tensor | float) -> torch.Tensor:
        position = position_w.to(device=self.device, dtype=torch.float32)
        if position.ndim == 1:
            position = position.unsqueeze(0)
        time_tensor = torch.as_tensor(time_s, device=self.device, dtype=torch.float32)
        if time_tensor.ndim == 0:
            time_tensor = time_tensor.repeat(position.shape[0])
        time_tensor = time_tensor.reshape(-1, 1, 1)

        x = position[:, 0].reshape(-1, 1, 1)
        y = position[:, 1].reshape(-1, 1, 1)
        z = torch.clamp(position[:, 2].reshape(-1, 1, 1).abs(), min=0.0, max=float(self.depth))

        phase = self.kx.unsqueeze(0) * x + self.ky.unsqueeze(0) * y - self.omega.unsqueeze(0) * time_tensor + self.phase.unsqueeze(0)
        depth_decay = torch.exp(-self.k.unsqueeze(0) * z)
        coeff = self.amplitude.unsqueeze(0) * self.omega.unsqueeze(0) * depth_decay * torch.cos(phase)
        u = torch.sum(coeff * torch.cos(self.directions).unsqueeze(0), dim=(1, 2))
        v = torch.sum(coeff * torch.sin(self.directions).unsqueeze(0), dim=(1, 2))
        w = torch.zeros_like(u)
        return torch.stack((u, v, w), dim=-1)

    def get_surface_elevation(self, position_xy: torch.Tensor, time_s: torch.Tensor | float) -> torch.Tensor:
        position = position_xy.to(device=self.device, dtype=torch.float32)
        if position.ndim == 1:
            position = position.unsqueeze(0)

        time_tensor = torch.as_tensor(time_s, device=self.device, dtype=torch.float32)
        if time_tensor.ndim == 0:
            time_tensor = time_tensor.repeat(position.shape[0])
        time_tensor = time_tensor.reshape(-1, 1, 1)

        x = position[:, 0].reshape(-1, 1, 1)
        y = position[:, 1].reshape(-1, 1, 1)
        phase = self.kx.unsqueeze(0) * x + self.ky.unsqueeze(0) * y - self.omega.unsqueeze(0) * time_tensor + self.phase.unsqueeze(0)
        return torch.sum(self.amplitude.unsqueeze(0) * torch.cos(phase), dim=(1, 2))

    def sample_horizontal_field(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        z_level: float,
        time_s: float,
        num_x: int = 41,
        num_y: int = 41,
    ) -> dict[str, torch.Tensor]:
        x = torch.linspace(float(x_limits[0]), float(x_limits[1]), int(num_x), device=self.device)
        y = torch.linspace(float(y_limits[0]), float(y_limits[1]), int(num_y), device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack(
            (xx.reshape(-1), yy.reshape(-1), torch.full((xx.numel(),), float(z_level), device=self.device)),
            dim=-1,
        )
        velocity = self.get_wave_velocity(positions, time_s).reshape(int(num_y), int(num_x), 3)
        surface = self.get_surface_elevation(torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1), time_s).reshape(
            int(num_y), int(num_x)
        )
        return {
            "x": xx.detach().cpu(),
            "y": yy.detach().cpu(),
            "surface": surface.detach().cpu(),
            "velocity": velocity.detach().cpu(),
        }

    def sample_vertical_slice(
        self,
        x_limits: tuple[float, float],
        z_limits: tuple[float, float],
        time_s: float,
        y_value: float = 0.0,
        num_x: int = 41,
        num_z: int = 21,
    ) -> dict[str, torch.Tensor]:
        x = torch.linspace(float(x_limits[0]), float(x_limits[1]), int(num_x), device=self.device)
        z = torch.linspace(float(z_limits[0]), float(z_limits[1]), int(num_z), device=self.device)
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        positions = torch.stack(
            (xx.reshape(-1), torch.full((xx.numel(),), float(y_value), device=self.device), zz.reshape(-1)),
            dim=-1,
        )
        velocity = self.get_wave_velocity(positions, time_s).reshape(int(num_z), int(num_x), 3)
        surface = self.get_surface_elevation(
            torch.stack((x, torch.full_like(x, float(y_value), device=self.device)), dim=-1),
            time_s,
        )
        return {
            "x": xx.detach().cpu(),
            "z": zz.detach().cpu(),
            "surface_x": x.detach().cpu(),
            "surface": surface.detach().cpu(),
            "velocity": velocity.detach().cpu(),
        }
