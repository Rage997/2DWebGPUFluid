// Numerical scheme adapted from Pavel Dobryakov's MIT-licensed WebGL Fluid Simulation.
// Uniforms occupy 96 bytes; each pass gets a separate 256-byte-aligned slot.
const uniforms = `
struct Uniforms {
  texelSize: vec2f, dyeTexelSize: vec2f,
  dt: f32, dissipation: f32, aspect: f32, curl: f32,
  pressure: f32, count: f32, mode: f32,
  direction: vec2f, background: vec4f, params: vec4f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
`;
// Texture coordinates use WebGPU's top-left origin throughout the solver.
const vertex = `
struct VSOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var o: VSOut; o.position=vec4f(p[i],0,1); o.uv=p[i]*vec2f(0.5,-0.5)+0.5; return o;
}
`;
const render = (source: string): string => uniforms + vertex + source;
const load = `
fn load(t: texture_2d<f32>, p: vec2i) -> vec4f {
  return textureLoad(t, clamp(p, vec2i(0), vec2i(textureDimensions(t))-1), 0);
}
`;
const compute = (bindings: string, body: string): string => uniforms + load + bindings + `
@compute @workgroup_size(8,8) fn cs(@builtin(global_invocation_id) id: vec3u) {
  let p=vec2i(id.xy);
  if (any(id.xy >= textureDimensions(outputTex)) || u.mode < -1.0) { return; }
  ${body}
}
`;
const output = (binding: number): string => `@group(0) @binding(${binding}) var outputTex: texture_storage_2d<rgba16float,write>;`;
export const SHADERS = {
  copy: vertex + `
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f { return textureSample(source,s,i.uv); }
`,
  splat: render(`
struct Splat { position: vec2f, delta: vec2f, color: vec3f };
@group(0) @binding(1) var<storage,read> splats: array<Splat>;
@group(0) @binding(2) var source: texture_2d<f32>;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 var value=textureLoad(source,vec2i(i.position.xy),0).rgb;
 for(var n=0u;n<u32(u.count);n++) {
   let sp=splats[n];
   var p=i.uv-vec2f(sp.position.x,1-sp.position.y); p.x*=u.aspect;
   let c=select(vec3f(sp.delta.x,-sp.delta.y,0),sp.color,u.mode>0.5);
   value+=exp(-dot(p,p)/max(u.params.x,0.000001))*c;
 }
 return vec4f(value,1);
}`),
  advection: render(`
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var vsampler: sampler;
@group(0) @binding(3) var source: texture_2d<f32>;
@group(0) @binding(4) var ssampler: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 let uv=i.uv-u.dt*textureSample(velocity,vsampler,i.uv).xy*u.texelSize;
 return vec4f(textureSample(source,ssampler,uv).rgb/(1+u.dt*u.dissipation),1);
}`),
  curl: compute(`@group(0) @binding(1) var velocity: texture_2d<f32>;`+output(2),`
 let L=load(velocity,p-vec2i(1,0)).y; let R=load(velocity,p+vec2i(1,0)).y;
 let B=load(velocity,p-vec2i(0,1)).x; let T=load(velocity,p+vec2i(0,1)).x;
 textureStore(outputTex,p,vec4f(0.5*(R-L-T+B),0,0,1));`),
  divergence: compute(`@group(0) @binding(1) var velocity: texture_2d<f32>;`+output(2),`
 let dims=vec2i(textureDimensions(velocity)); let C=load(velocity,p).xy;
 var L=load(velocity,p-vec2i(1,0)).x; var R=load(velocity,p+vec2i(1,0)).x;
 var B=load(velocity,p-vec2i(0,1)).y; var T=load(velocity,p+vec2i(0,1)).y;
 if(p.x==0){L=-C.x;} if(p.x==dims.x-1){R=-C.x;}
 if(p.y==0){B=-C.y;} if(p.y==dims.y-1){T=-C.y;}
 textureStore(outputTex,p,vec4f(0.5*(R-L+T-B),0,0,1));`),
  vorticity: compute(`
@group(0) @binding(1) var curl: texture_2d<f32>;
@group(0) @binding(2) var velocity: texture_2d<f32>;`+output(3),`
 let L=load(curl,p-vec2i(1,0)).x; let R=load(curl,p+vec2i(1,0)).x;
 let B=load(curl,p-vec2i(0,1)).x; let T=load(curl,p+vec2i(0,1)).x;
 var f=0.5*vec2f(abs(T)-abs(B),abs(L)-abs(R));
 f=f/(length(f)+0.0001)*u.curl*load(curl,p).x;
 textureStore(outputTex,p,vec4f(load(velocity,p).xy+f*u.dt,0,1));`),
  pressureClear: compute(`@group(0) @binding(1) var pressure: texture_2d<f32>;`+output(2),`
 textureStore(outputTex,p,vec4f(load(pressure,p).x*u.pressure,0,0,1));`),
  jacobi: compute(`
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var divergence: texture_2d<f32>;`+output(3),`
 let L=load(pressure,p-vec2i(1,0)).x; let R=load(pressure,p+vec2i(1,0)).x;
 let B=load(pressure,p-vec2i(0,1)).x; let T=load(pressure,p+vec2i(0,1)).x;
 textureStore(outputTex,p,vec4f((L+R+B+T-load(divergence,p).x)*0.25,0,0,1));`),
  gradientSubtract: compute(`
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var velocity: texture_2d<f32>;`+output(3),`
 let L=load(pressure,p-vec2i(1,0)).x; let R=load(pressure,p+vec2i(1,0)).x;
 let B=load(pressure,p-vec2i(0,1)).x; let T=load(pressure,p+vec2i(0,1)).x;
 textureStore(outputTex,p,vec4f(load(velocity,p).xy-vec2f(R-L,T-B),0,1));`),
  display: render(`
@group(0) @binding(1) var dye: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 var c=textureSample(dye,s,i.uv).rgb;
 if(u.params.x>0.5){
 let L=length(textureSample(dye,s,i.uv-vec2f(u.dyeTexelSize.x,0)).rgb);
 let R=length(textureSample(dye,s,i.uv+vec2f(u.dyeTexelSize.x,0)).rgb);
 let B=length(textureSample(dye,s,i.uv-vec2f(0,u.dyeTexelSize.y)).rgb);
 let T=length(textureSample(dye,s,i.uv+vec2f(0,u.dyeTexelSize.y)).rgb);
 let n=normalize(vec3f(R-L,T-B,length(u.dyeTexelSize)));
 c*=clamp(n.z+0.7,0.7,1.0);
 }
 return vec4f(c,1);
}`),
  bloomPrefilter: render(`
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 let c=textureSample(source,s,i.uv).rgb; let br=max(c.r,max(c.g,c.b));
 let knee=u.params.x*0.7+0.0001; var soft=clamp(br-u.params.x+knee,0,2*knee);
 soft=soft*soft/(4*knee);
 return vec4f(c*max(soft,br-u.params.x)/max(br,0.0001),0);
}`),
  bloomBlur: render(`
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 let x=vec2f(u.texelSize.x,0); let y=vec2f(0,u.texelSize.y);
 return (textureSample(source,s,i.uv-x)+textureSample(source,s,i.uv+x)+textureSample(source,s,i.uv-y)+textureSample(source,s,i.uv+y))*0.25;
}`),
  sunrays: render(`
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 var uv=i.uv; let delta=(uv-0.5)*0.3/16; var decay=1.0; var value=0.0;
 for(var n=0;n<16;n++){
 uv-=delta; let c=textureSampleLevel(source,s,uv,0).rgb;
 let mask=1-clamp(max(c.r,max(c.g,c.b))*20,0,0.8);
 value+=mask*decay*u.params.y; decay*=0.95;
 }
 return vec4f(value*0.7,0,0,1);
}`),
  composite: render(`
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@group(0) @binding(3) var bloom: texture_2d<f32>;
@group(0) @binding(4) var bs: sampler;
@group(0) @binding(5) var rays: texture_2d<f32>;
@group(0) @binding(6) var rs: sampler;
@fragment fn fs(i: VSOut)->@location(0) vec4f {
 var c=textureSample(source,s,i.uv).rgb;
 var glow=textureSample(bloom,bs,i.uv).rgb*u.params.x;
 if(u.params.y>0){ let ray=textureSample(rays,rs,i.uv).r; c*=ray; glow*=ray; }
 if(u.params.x>0){
 let noise=fract(sin(dot(i.position.xy,vec2f(12.9898,78.233)))*43758.5453)-0.5;
 glow=max(glow+noise/255,vec3f(0));
 c+=max(1.055*pow(glow,vec3f(1.0/2.4))-0.055,vec3f(0));
 }
 let a=clamp(max(c.r,max(c.g,c.b)),0,1);
 c=clamp(c,vec3f(0),vec3f(1));
 if(u.params.z>0.5){
   if(u.params.w>0.5){return vec4f(c/max(a,0.00001),a);}
   return vec4f(c,a);
 }
 return vec4f(c+u.background.rgb*(1-a),1);
}`),
};
