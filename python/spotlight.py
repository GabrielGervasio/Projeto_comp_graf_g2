from __future__ import annotations

import math
from typing import Any

import graphics_math as gm

from light import Light


class SpotLight(Light):
    """Spot light whose position and direction map to the shader's light space."""

    def __init__ (self, x: float, y: float, z: float,
                  dx: float, dy: float, dz: float, theta: float,
                  exponent: float = 1.0, space: str = "world",
                  **values: Any) -> None:
      Light.__init__(self, space, **values)
      self.set_position(x, y, z)
      self.set_direction(dx, dy, dz)
      self.set_angle(theta)
      self.set_exponent(exponent)

    def set_position (self, x: float, y: float, z: float) -> None:
      self.set("light_position", gm.vec4(x, y, z, 1.0))

    def set_direction (self, x: float, y: float, z: float) -> None:
      if not all(math.isfinite(component) for component in (x, y, z)):
        raise ValueError("Spot direction components must be finite")
      if x == 0.0 and y == 0.0 and z == 0.0:
        raise ValueError("Spot direction must be non-zero")
      self.set("spot_direction", gm.vec3(x, y, z))

    def set_angle (self, theta: float) -> None:
      if not math.isfinite(theta) or not 0.0 < theta < 180.0:
        raise ValueError("Spot angle must be between 0 and 180 degrees")
      self.set("spot_cutoff", math.cos(math.radians(theta)))

    def set_exponent (self, exponent: float) -> None:
      if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("Spot exponent must be finite and non-negative")
      self.set("spot_exponent", exponent)

    def map (self, matrix: gm.Mat4) -> dict[str, Any]:
      position = matrix @ self.get("light_position")
      direction = self.get("spot_direction")
      direction = matrix @ gm.vec4(direction[0], direction[1], direction[2], 0.0)
      return {
        "light_position": position,
        "spot_direction": gm.normalize(direction[:3]),
      }
