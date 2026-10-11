// Orbit camera for 3D phenomena: slowly circles a target; drag to turn it.

export type Mat4 = Float32Array<ArrayBuffer>;

function normalize(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(...v) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: number[], b: number[]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Column-major view-projection matrix (WebGPU clip space, z in [0, 1]). */
export function lookAtPerspective(
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number],
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): Mat4 {
  const f = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const s = normalize(cross(f, up));
  const u = cross(s, f);
  const t = 1 / Math.tan(fovY / 2);
  const a = far / (near - far);
  const b = (near * far) / (near - far);
  // P * V written out directly.
  const ex = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
  const ey = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
  const ez = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
  const m = new Float32Array(16);
  for (let c = 0; c < 3; c++) {
    m[c * 4 + 0] = (t / aspect) * s[c];
    m[c * 4 + 1] = t * u[c];
    m[c * 4 + 2] = -a * f[c];
    m[c * 4 + 3] = f[c];
  }
  m[12] = (t / aspect) * ex;
  m[13] = t * ey;
  m[14] = a * ez + b;
  m[15] = -ez;
  return m;
}

export class OrbitCamera {
  yaw = 0.6;
  pitch = 0.18;
  distance = 1;
  private spin = 0;
  private yawVel = 0;
  private pitchVel = 0;

  constructor(
    public target: [number, number, number],
    public baseDistance: number,
    public autoRotate = 0.035,
  ) {
    this.distance = baseDistance;
  }

  drag(dx: number, dy: number, dt: number): void {
    const k = 0.004;
    this.yaw -= dx * k;
    this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + dy * k));
    const a = 1 - Math.exp(-dt * 30);
    this.yawVel += ((-dx * k) / Math.max(dt, 1e-3) - this.yawVel) * a;
    this.pitchVel += ((dy * k) / Math.max(dt, 1e-3) - this.pitchVel) * a;
  }

  zoom(steps: number): void {
    this.distance = Math.max(this.baseDistance * 0.35, Math.min(this.baseDistance * 2.5, this.distance * 2 ** (-steps * 0.18)));
  }

  update(dt: number, dragging: boolean, paused: boolean): void {
    if (!dragging) {
      this.yaw += this.yawVel * dt;
      this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + this.pitchVel * dt));
      const k = Math.exp(-dt * 2.5);
      this.yawVel *= k;
      this.pitchVel *= k;
      // Auto-rotation eases back in after the viewer lets go.
      this.spin = Math.min(1, this.spin + dt * 0.3);
      if (!paused) this.yaw += this.autoRotate * this.spin * dt;
    } else {
      this.spin = 0;
    }
  }

  eye(): [number, number, number] {
    const [tx, ty, tz] = this.target;
    const cp = Math.cos(this.pitch);
    return [tx + this.distance * cp * Math.cos(this.yaw), ty + this.distance * cp * Math.sin(this.yaw), tz + this.distance * Math.sin(this.pitch)];
  }

  matrix(aspect: number): Mat4 {
    return lookAtPerspective(this.eye(), this.target, [0, 0, 1], (38 * Math.PI) / 180, aspect, 1, 1000);
  }
}
