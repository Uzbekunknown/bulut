#version 450
// test_vk_sim's scene: a panorama around a camera at `yaw` (turned right) and `pitch` (looking
// down), focal f; a red stripe at world yaw 0 and a checker of 10-degree cells elsewhere, depth
// 0.9 (calib: a non-periodic noise instead, for the calibration fit); and a grey box at the bottom
// right, "the weapon", at depth 0.05.
// calib 2: a cylinder of radius 4 around the origin, the camera at (px, 0, pz) looking along +z,
// depth 1 - 0.1 / z (near plane 0.1, far at infinity): parallax when the camera moves
layout(push_constant) uniform P { float yaw, pitch, f, W, H, calib, px, pz; } p;
layout(location = 0) out vec4 col;

float hash(vec2 c) { return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453); }

float vnoise(vec2 x)
{
	vec2 i = floor(x), fr = fract(x);
	vec2 u = fr * fr * (3.0 - 2.0 * fr);
	return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

void main()
{
	vec2 q = gl_FragCoord.xy;
	if (q.x >= 0.7 * p.W && q.x < 0.9 * p.W && q.y >= 0.7 * p.H && q.y < 0.95 * p.H) {
		col = vec4(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0);
		gl_FragDepth = 0.05;
		return;
	}
	vec3 d = vec3((q.x - 0.5 * p.W) / p.f, (q.y - 0.5 * p.H) / p.f, 1.0);
	if (p.calib > 1.5) {
		float a = d.x * d.x + 1.0, b = 2.0 * (p.px * d.x + p.pz), c = p.px * p.px + p.pz * p.pz - 16.0;
		float t = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);   // along d, so t is the view depth
		vec3 hit = vec3(p.px, 0.0, p.pz) + t * d;
		float th = atan(hit.x, hit.z);
		gl_FragDepth = 1.0 - 0.1 / t;
		if (abs(th) < 0.012) {
			col = vec4(1.0, 0.0, 0.0, 1.0);
		} else if (p.yaw > 0.5) {                                 // the walking calibration: a noise texture
			vec2 g = vec2(th * 4.0, hit.y);
			float nz = 0.45 * vnoise(g * 6.0) + 0.35 * vnoise(g * 15.0 + 7.0) + 0.2 * vnoise(g * 35.0 + 3.0);
			col = vec4(nz, 0.9 * nz, 0.8 * nz, 1.0);
		} else {
			float ch = mod(floor(th / 0.05) + floor(hit.y / 0.5), 2.0);
			col = vec4(0.1 + 0.5 * ch, 0.3, 0.6 - 0.4 * ch, 1.0);
		}
		return;
	}
	float cp = cos(p.pitch), sp = sin(p.pitch);
	vec3 l = vec3(d.x, d.y * cp + d.z * sp, -d.y * sp + d.z * cp);   // the ray before the pitch
	float a = atan(l.x, l.z) + p.yaw;
	float e = atan(-l.y, length(l.xz));
	gl_FragDepth = 0.9;
	if (p.calib > 0.5) {
		vec2 g = vec2(a, e);
		float n = 0.45 * vnoise(g * 15.0) + 0.35 * vnoise(g * 40.0 + 7.0) + 0.2 * vnoise(g * 90.0 + 3.0);
		col = vec4(n, 0.9 * n, 0.8 * n, 1.0);
		return;
	}
	if (abs(a) < 0.01) {
		col = vec4(1.0, 0.0, 0.0, 1.0);
	} else {
		float c = mod(floor(a / 0.1745) + floor(e / 0.1745), 2.0);
		col = vec4(0.1 + 0.5 * c, 0.3, 0.6 - 0.4 * c, 1.0);
	}
}
