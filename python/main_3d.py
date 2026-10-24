from __future__ import annotations

from pathlib import Path
from typing import Any

import graphics_math as gm
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera3d import *
from spotlight import *
from shader import *
from pipeline import *
from phongmaterial import *
from transform import *
from node import *
from scene import *
from renderer import *
from cube import *
from sphere import *
from appearance import Appearance
from clipplane import ClipPlane
from texture import Texture
from sampler import Sampler
from textureset import TextureSet

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera3D
scene: Scene
clip_plane: ClipPlane
clip_enabled = False
shader: Shader
second_light_enabled = True
second_light_ambient = gm.vec3(0.035, 0.035, 0.035)
second_light_diffuse = gm.vec3(0.28, 0.30, 0.34)
second_light_specular = gm.vec3(0.16, 0.16, 0.18)

viewer_pos: gm.Vec3 = gm.vec3(5.5, 7.3, 7.2)

def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene, clip_plane, clip_enabled, shader, second_light_enabled

  camera = Camera3D(viewer_pos[0],viewer_pos[1],viewer_pos[2])
  camera.set_center(0.0, 1.8, 0.0)
  camera.set_angle(40.0)
  arcball = camera.create_arcball()
  arcball.attach(canvas)

  light = SpotLight(
    -3.0, 9.0, 5.0,
    3.0, -6.76, -5.0,
    theta=45.0, exponent=18.0, space="world",
  )
  light.set_ambient(0.08, 0.08, 0.08)
  light.set_diffuse(0.95, 0.95, 0.95)

  tabletop_material = PhongMaterial(0.82, 0.84, 0.80)
  trim_material = PhongMaterial(0.035, 0.04, 0.04)
  green_material = PhongMaterial(0.08, 0.68, 0.16)
  green_material.set_specular(0.25, 0.25, 0.25)
  cube = Cube(device)
  sphere = Sphere(device)

  shader = Shader(device, "../shaders/ilum_frag/lit.wgsl", light=light, space="world")
  shader.set_value("second_light_direction", gm.normalize(gm.vec3(-0.4, 0.8, 0.3)))
  second_light_enabled = True
  set_second_light_enabled(second_light_enabled)
  shader.set_value("fog_color", gm.vec3(1.0, 1.0, 1.0))
  shader.set_value("fog_start", 7.0)
  shader.set_value("fog_end", 16.0)
  clip_plane = ClipPlane(shader, 0.0, 0.0, 0.0, 0.0)
  clip_enabled = False
  shader.set_vertex_buffers([
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ])
  pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "back"})

  for material in [tabletop_material, trim_material, green_material]:
    shader.add_material(material)

  decal_material = PhongMaterial(1.0, 1.0, 1.0)
  decal_material.set_specular(0.25, 0.25, 0.25)
  shader.add_material(decal_material)
  wood_texture_path = Path(__file__).resolve().parent.parent / "images" / "Wood021_4K-JPG_Color.jpg"
  decal_texture = Texture(device, "decal_texture", str(wood_texture_path))
  decal_sampler = Sampler(device, "decal_sampler")
  decal_set = TextureSet([decal_texture, decal_sampler])
  shader.add_texture_set(decal_set)
  ball_texture_path = Path(__file__).resolve().parent.parent / "images" / "bola" / "Wood020_4K-PNG_Color.png"
  ball_texture = Texture(device, "decal_texture", str(ball_texture_path))
  ball_decal_set = TextureSet([ball_texture, decal_sampler])
  shader.add_texture_set(ball_decal_set)
  default_texture = Texture(device, "decal_texture", None, texel=gm.vec4(1.0, 1.0, 1.0, 1.0))
  default_texture_set = TextureSet([default_texture, decal_sampler])
  shader.add_texture_set(default_texture_set)

  def block (material: PhongMaterial, position: tuple[float, float, float],
             size: tuple[float, float, float], decal: bool = False) -> Node:
    transform = Transform()
    transform.translate(*position)
    transform.scale(*size)
    apps: list[Appearance] = [material]
    if decal:
      apps.append(decal_set)
    return Node(None, transform, apps, [cube])

  def ball (material: PhongMaterial, position: tuple[float, float, float],
            radius: float, decal: bool = False) -> Node:
    transform = Transform()
    transform.translate(*position)
    transform.scale(radius, radius, radius)
    apps: list[Appearance] = [material]
    if decal:
      apps.append(ball_decal_set)
    return Node(None, transform, apps, [sphere])

  table_size = 5.2
  table_half = table_size / 2
  table_edge = table_half - 0.0175
  leg_offset = table_half - 0.5
  table_top = 2.24
  children = [
    block(tabletop_material, (0.0, 2.0, 0.0), (table_size, 0.24, table_size)),
    block(trim_material, (0.0, table_top, -table_edge), (table_size, 0.035, 0.035)),
    block(trim_material, (0.0, table_top, table_edge), (table_size, 0.035, 0.035)),
    block(trim_material, (-table_edge, table_top, 0.0), (0.035, 0.035, table_size)),
    block(trim_material, (table_edge, table_top, 0.0), (0.035, 0.035, table_size)),
    block(trim_material, (-leg_offset, 0.0, -leg_offset), (0.22, 2.0, 0.22)),
    block(trim_material, (leg_offset, 0.0, -leg_offset), (0.22, 2.0, 0.22)),
    block(trim_material, (-leg_offset, 0.0, leg_offset), (0.22, 2.0, 0.22)),
    block(trim_material, (leg_offset, 0.0, leg_offset), (0.22, 2.0, 0.22)),
    block(decal_material, (-0.72, table_top, 0.05), (1.35, 0.6, 1.85), decal=True),
    ball(green_material, (-0.72, table_top + 0.6 + 0.31, 0.05), 0.31),
    ball(decal_material, (1.35, table_top + 0.6, -0.55), 0.6, decal=True),
  ]
  scene = Scene(Node(pipeline, apps=[default_texture_set], nodes=children))

def draw () -> None:
  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)

def set_second_light_enabled (enabled: bool) -> None:
  values = (second_light_ambient, second_light_diffuse, second_light_specular) if enabled else (
    gm.vec3(0.0, 0.0, 0.0),
    gm.vec3(0.0, 0.0, 0.0),
    gm.vec3(0.0, 0.0, 0.0),
  )
  shader.set_value("second_light_ambient", values[0])
  shader.set_value("second_light_diffuse", values[1])
  shader.set_value("second_light_specular", values[2])

def on_key (event: Any) -> None:
  global clip_enabled, second_light_enabled
  if event["key"] == "q":
    canvas.close()
  elif event["key"].lower() == "c":
    clip_enabled = not clip_enabled
    if clip_enabled:
      clip_plane.set_plane(0.0, 0.0, 1.0, 0.0)
    else:
      clip_plane.set_plane(0.0, 0.0, 0.0, 0.0)
  elif event["key"].lower() == "l":
    second_light_enabled = not second_light_enabled
    set_second_light_enabled(second_light_enabled)

def main () -> None:
  global canvas, device, context, renderer

  canvas = RenderCanvas(size=(640, 480), title="3D scene", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  # formato preferido COM "-srgb": a GPU codifica de linear para sRGB
  # automaticamente na saída — correto desde que o shader faça a conta de
  # iluminação em espaço linear.
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=(1.0, 1.0, 1.0, 1.0))

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  canvas.request_draw(draw)
  loop.run()

if __name__ == "__main__":
  main()
