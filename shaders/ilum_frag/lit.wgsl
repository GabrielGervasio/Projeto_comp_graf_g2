struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,   // espaco de iluminacao -> NDC
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  spot_direction: vec3<f32>,
  spot_cutoff: f32,
  spot_exponent: f32,
  camera_position: vec4<f32>,  // posição da câmera no espaço de iluminação (não usada nesta fórmula)
  second_light_direction: vec3<f32>,
  second_light_ambient: vec3<f32>,
  second_light_diffuse: vec3<f32>,
  second_light_specular: vec3<f32>,
  fog_color: vec3<f32>,
  fog_start: f32,
  fog_end: f32,
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) position: vec3<f32>,
  @location(2) view: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
}

// Iluminacao por fragmento: posicao e normal sao interpoladas para que
// direcao da luz e coeficiente do spot sejam calculados por pixel.
@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.position = veye;
  out.view = normalize(-veye);
  out.texcoord = texcoord;
  out.clip_position = global.projection * (m.vertex * vec4<f32>(coord, 1.0));
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  // renormaliza: a interpolacao entre vertices desnormaliza normal/luz/vista
  if (dot(global.clip_plane, vec4<f32>(in.position, 1.0)) < 0.0) {
    discard;
  }

  let normal = normalize(in.normal);
  let light = normalize(global.light_position.xyz - in.position);
  let view = normalize(in.view);
  let spot_cosine = dot(-light, normalize(global.spot_direction));
  let spot_factor = step(global.spot_cutoff, spot_cosine)
    * pow(max(spot_cosine, 0.0), global.spot_exponent);

  var color = material.base_color * global.light_ambient.rgb * spot_factor;
  let ndotl = dot(normal, light);
  if (ndotl > 0.0) {
    color += material.base_color * global.light_diffuse.rgb * ndotl * spot_factor;
    let refl = normalize(reflect(-light, normal));
    color += material.specular_color * global.light_specular.rgb
      * pow(max(0.0, dot(refl, view)), material.shininess) * spot_factor;
  }

  let second_light = normalize(global.second_light_direction);
  let second_ndotl = max(dot(normal, second_light), 0.0);
  color += material.base_color * global.second_light_ambient;
  color += material.base_color * global.second_light_diffuse * second_ndotl;
  if (second_ndotl > 0.0) {
    let second_refl = normalize(reflect(-second_light, normal));
    color += material.specular_color * global.second_light_specular
      * pow(max(0.0, dot(second_refl, view)), material.shininess);
  }

  let decal = textureSample(decal_texture, decal_sampler, in.texcoord);
  let lit_color = color * decal.rgb;
  let fog_amount = clamp(
    (distance(in.position, global.camera_position.xyz) - global.fog_start)
      / (global.fog_end - global.fog_start),
    0.0, 1.0,
  );
  return vec4<f32>(mix(lit_color, global.fog_color, fog_amount), decal.a);
}
