use crate::accel::AxisBB;
use crate::material::Isotropic;
use crate::material::{MaterialSS, TextureSS};
use crate::rand::Rng;
use crate::util::{fmax, fmin};
use crate::vec3::Vec3;
use crate::Ray;
use std::sync::Arc;

#[derive(new)]
pub struct HitRec {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub front: bool,
    pub material: Arc<MaterialSS>,
}

impl HitRec {
    fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front = r.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

pub trait Hittable {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec>;
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB>;
}

pub type HittableSS = dyn Hittable + Send + Sync;

#[derive(new)]
pub struct Sphere {
    center: Vec3,
    radius: f32,
    material: Arc<MaterialSS>,
}

impl Sphere {
    pub fn spherical(p: Vec3) -> (f32, f32) {
        let pi = std::f32::consts::PI;
        let phi = p.z.atan2(p.x);
        let theta = p.y.asin();
        let u = 1.0 - ((phi + pi) / (2.0 * pi));
        let v = (theta + pi / 2.0) / pi;
        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let oc = r.origin - self.center;
        let a = r.direction.length2();
        let half_b = oc.dot(r.direction);
        let c = oc.length2() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            for temp in &[(-half_b - root) / a, (-half_b + root) / a] {
                if tmin < *temp && *temp < tmax {
                    let mut ret = HitRec::new(
                        r.at(*temp),
                        (r.at(*temp) - self.center) / self.radius,
                        *temp,
                        0.0,
                        0.0,
                        false,
                        self.material.clone(),
                    );
                    ret.set_face_normal(r, (ret.p - self.center) / self.radius);
                    let (u, v) = Sphere::spherical((ret.p - self.center) / self.radius);
                    ret.u = u;
                    ret.v = v;
                    return Some(ret);
                }
            }
        }

        None
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        Some(AxisBB::new(
            self.center - Vec3::new_const(self.radius),
            self.center + Vec3::new_const(self.radius),
        ))
    }
}

#[derive(new)]
pub struct MovingSphere {
    center0: Vec3,
    center1: Vec3,
    time0: f32,
    time1: f32,
    radius: f32,
    material: Arc<MaterialSS>,
}

impl MovingSphere {
    pub fn center(&self, time: f32) -> Vec3 {
        self.center0
            + (self.center1 - self.center0) * ((time - self.time0) / (self.time1 - self.time0))
    }
}

impl Hittable for MovingSphere {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let oc = r.origin - self.center(r.time);
        let a = r.direction.length2();
        let half_b = oc.dot(r.direction);
        let c = oc.length2() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            for temp in &[(-half_b - root) / a, (-half_b + root) / a] {
                if tmin < *temp && *temp < tmax {
                    let mut ret = HitRec::new(
                        r.at(*temp),
                        (r.at(*temp) - self.center(r.time)) / self.radius,
                        *temp,
                        0.0,
                        0.0,
                        false,
                        self.material.clone(),
                    );
                    ret.set_face_normal(r, (ret.p - self.center(r.time)) / self.radius);
                    let (u, v) = Sphere::spherical((ret.p - self.center(r.time)) / self.radius);
                    ret.u = u;
                    ret.v = v;
                    return Some(ret);
                }
            }
        }

        None
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        let bb1 = AxisBB::new(
            self.center(self.time0) - Vec3::new_const(self.radius),
            self.center(self.time0) + Vec3::new_const(self.radius),
        );
        let bb2 = AxisBB::new(
            self.center(self.time1) - Vec3::new_const(self.radius),
            self.center(self.time1) + Vec3::new_const(self.radius),
        );
        Some(AxisBB::surrounding_box(bb1, bb2))
    }
}

#[derive(new)]
pub struct Rect {
    c0: f32,
    c1: f32,
    d0: f32,
    d1: f32,
    k: f32,
    axis0: usize,
    axis1: usize,
    axis2: usize,
    mat: Arc<MaterialSS>,
}

impl Rect {
    #[allow(non_snake_case)]
    pub fn XYRect(x0: f32, x1: f32, y0: f32, y1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(x0, x1, y0, y1, k, 0, 1, 2, mat)
    }

    #[allow(non_snake_case)]
    pub fn XZRect(x0: f32, x1: f32, z0: f32, z1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(x0, x1, z0, z1, k, 0, 2, 1, mat)
    }

    #[allow(non_snake_case)]
    pub fn YZRect(y0: f32, y1: f32, z0: f32, z1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(y0, y1, z0, z1, k, 1, 2, 0, mat)
    }
}

impl Hittable for Rect {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let t = (self.k - r.origin[self.axis2]) / r.direction[self.axis2];
        if t < tmin || t > tmax {
            return None;
        }
        let a = r.origin[self.axis0] + t * r.direction[self.axis0];
        let b = r.origin[self.axis1] + t * r.direction[self.axis1];
        if a < self.c0 || a > self.c1 || b < self.d0 || b > self.d1 {
            return None;
        }
        let mut outward_normal = Vec3::new_const(0.0);
        outward_normal[self.axis2] = 1.0;
        let u = (a - self.c0) / (self.c1 - self.c0);
        let v = (b - self.d0) / (self.d1 - self.d0);

        let mut ret = HitRec::new(
            r.at(t),
            Vec3::new_const(0.0), // overwritten later
            t,
            u,
            v,
            false, // overwritten later
            self.mat.clone(),
        );
        ret.set_face_normal(r, outward_normal);
        Some(ret)
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        // Make the box a little thicc to prevent zero division
        let mut v1 = Vec3::new_const(0.0);
        let mut v2 = Vec3::new_const(0.0);
        v1[self.axis0] = self.c0;
        v1[self.axis1] = self.d0;
        v1[self.axis2] = self.k - 0.0001;
        v2[self.axis0] = self.c1;
        v2[self.axis1] = self.d1;
        v2[self.axis2] = self.k + 0.0001;
        Some(AxisBB::new(v1, v2))
    }
}

#[derive(new)]
pub struct FlipFace {
    ptr: Arc<HittableSS>,
}

impl Hittable for FlipFace {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        if let Some(h) = self.ptr.hit(r, tmin, tmax) {
            Some(HitRec::new(
                h.p, h.normal, h.t, h.u, h.v, !h.front, h.material,
            ))
        } else {
            None
        }
    }
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        self.ptr.bounding_box(t0, t1)
    }
}

pub struct Boxy {
    box_min: Vec3,
    box_max: Vec3,
    sides: Vec<Arc<HittableSS>>,
}

impl Boxy {
    pub fn new(p0: Vec3, p1: Vec3, mat: Arc<MaterialSS>) -> Boxy {
        assert!(p0.x < p1.x);
        assert!(p0.y < p1.y);
        assert!(p0.z < p1.z);
        let sides: Vec<Arc<HittableSS>> = vec![
            Arc::new(Rect::XYRect(p0.x, p1.x, p0.y, p1.y, p1.z, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::XYRect(
                p0.x,
                p1.x,
                p0.y,
                p1.y,
                p0.z,
                mat.clone(),
            )))),
            Arc::new(Rect::XZRect(p0.x, p1.x, p0.z, p1.z, p1.y, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::XZRect(
                p0.x,
                p1.x,
                p0.z,
                p1.z,
                p0.y,
                mat.clone(),
            )))),
            Arc::new(Rect::YZRect(p0.y, p1.y, p0.z, p1.z, p1.x, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::YZRect(
                p0.y,
                p1.y,
                p0.z,
                p1.z,
                p0.x,
                mat.clone(),
            )))),
        ];
        Boxy {
            box_min: p0,
            box_max: p1,
            sides,
        }
    }
}

impl Hittable for Boxy {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        self.sides.hit(r, tmin, tmax)
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        Some(AxisBB::new(self.box_min, self.box_max))
    }
}

impl Hittable for Vec<Arc<HittableSS>> {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let mut closest_dist = tmax;
        let mut closest_rec: Option<HitRec> = None;
        for w in self {
            if let Some(rec) = w.hit(&r, tmin, closest_dist) {
                if rec.t < closest_dist {
                    closest_dist = rec.t;
                    closest_rec = Some(rec);
                }
            }
        }

        closest_rec
    }

    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        if self.is_empty() {
            return None;
        }

        let mut output_box: Option<AxisBB> = None;
        let mut first_box = true;

        for o in self {
            if let Some(bb) = o.bounding_box(t0, t1) {
                if first_box {
                    output_box = Some(bb);
                } else {
                    output_box = Some(AxisBB::surrounding_box(bb, output_box.unwrap()));
                }
            } else {
                return None;
            }
            first_box = false;
        }

        output_box
    }
}

pub struct ConstantMedium {
    boundary: Arc<HittableSS>,
    phase_function: Arc<MaterialSS>,
    neg_inv_density: f32,
}

impl ConstantMedium {
    pub fn new(boundary: Arc<HittableSS>, density: f32, albedo: Arc<TextureSS>) -> ConstantMedium {
        ConstantMedium {
            boundary,
            phase_function: Arc::new(Isotropic::new(albedo.clone())),
            neg_inv_density: -1.0 / density,
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let mut rng = rand::thread_rng();
        // rec1 and rec2 here define the near and far points where
        // the ray hits the boundary of the object
        if let Some(mut rec1) = self.boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY) {
            if let Some(mut rec2) = self.boundary.hit(r, rec1.t + 0.0001, f32::INFINITY) {
                if rec1.t < tmin {
                    rec1.t = tmin
                }
                if rec2.t > tmax {
                    rec2.t = tmax
                }
                if rec1.t >= rec2.t {
                    return None;
                }
                if rec1.t < 0.0 {
                    rec1.t = 0.0;
                }
                let ray_length = r.direction.length();
                let distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
                let hit_distance = self.neg_inv_density * rng.gen::<f32>().ln();
                if hit_distance > distance_inside_boundary {
                    return None;
                }

                // Construct the output
                let t = rec1.t + hit_distance / ray_length;

                return Some(HitRec::new(
                    r.at(t),
                    Vec3::new(1.0, 0.0, 0.0), // arbitrary
                    t,
                    rec1.u,
                    rec1.v, // aribtrary
                    true,   // arbitrary
                    self.phase_function.clone(),
                ));
            }
        }
        None
    }

    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        self.boundary.bounding_box(t0, t1)
    }
}

#[derive(new)]
pub struct Translate {
    ptr: Arc<HittableSS>,
    offset: Vec3,
}

impl Hittable for Translate {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let moved_r = Ray::new_with_time(r.origin - self.offset, r.direction, r.time);
        if let Some(rec) = self.ptr.hit(&moved_r, tmin, tmax) {
            let mut moved_rec = HitRec::new(
                rec.p + self.offset,
                rec.normal,
                rec.t,
                rec.u,
                rec.v,
                rec.front,
                rec.material,
            );
            moved_rec.set_face_normal(&moved_r, rec.normal);
            Some(moved_rec)
        } else {
            None
        }
    }
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        if let Some(bb) = self.ptr.bounding_box(t0, t1) {
            Some(AxisBB::new(bb.min + self.offset, bb.max + self.offset))
        } else {
            None
        }
    }
}

pub struct RotateY {
    ptr: Arc<HittableSS>,
    sin_theta: f32,
    cos_theta: f32,
    bb: Option<AxisBB>,
}

impl RotateY {
    pub fn new(p: Arc<HittableSS>, angle: f32) -> RotateY {
        let sin_theta = angle.to_radians().sin();
        let cos_theta = angle.to_radians().cos();

        let bbox = p.bounding_box(0.0, 1.0).unwrap();

        let mut min = Vec3::new_const(f32::INFINITY);
        let mut max = Vec3::new_const(f32::NEG_INFINITY);

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let x = if i == 1 { bbox.max.x } else { bbox.min.x };
                    let y = if j == 1 { bbox.max.y } else { bbox.min.y };
                    let z = if k == 1 { bbox.max.z } else { bbox.min.z };

                    let new_x = cos_theta * x + sin_theta * z;
                    let new_z = -sin_theta * x + cos_theta * z;
                    let tester = Vec3::new(new_x, y, new_z);
                    for c in 0..3 {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        RotateY {
            ptr: p,
            sin_theta,
            cos_theta,
            bb: Some(AxisBB::new(min, max)),
        }
    }
}

impl Hittable for RotateY {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let mut origin = r.origin;
        let mut direction = r.direction;

        // Note that origin/direction are in world coordinates and
        // need to be transformed into object coords to do hit check.
        // This is the inverse of the transformation done when computing
        // the normal below, and relies on the fact that the inverse of
        // a rotation by t is a rotation by -t, and
        //       cos(-t) = cos(t)
        //       sin(-t) = -sin(t)
        // See https://github.com/RayTracing/raytracing.github.io/issues/544
        origin.x = self.cos_theta * r.origin.x - self.sin_theta * r.origin.z;
        origin.z = self.sin_theta * r.origin.x + self.cos_theta * r.origin.z;

        direction.x = self.cos_theta * r.direction.x - self.sin_theta * r.direction.z;
        direction.z = self.sin_theta * r.direction.x + self.cos_theta * r.direction.z;

        let rotated_r = Ray::new_with_time(origin, direction, r.time);

        if let Some(rec) = self.ptr.hit(&rotated_r, tmin, tmax) {
            let mut p = rec.p;
            let mut normal = rec.normal;

            p.x = self.cos_theta * rec.p.x + self.sin_theta * rec.p.z;
            p.z = -self.sin_theta * rec.p.x + self.cos_theta * rec.p.z;

            normal.x = self.cos_theta * rec.normal.x + self.sin_theta * rec.normal.z;
            normal.z = -self.sin_theta * rec.normal.x + self.cos_theta * rec.normal.z;

            let mut out_rec = HitRec::new(
                p,
                normal,
                rec.t,
                rec.u,
                rec.v,
                rec.front,
                rec.material.clone(),
            );
            out_rec.set_face_normal(&rotated_r, normal);

            return Some(out_rec);
        }

        None
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        self.bb
    }
}

pub struct RotateX {
    ptr: Arc<HittableSS>,
    sin_theta: f32,
    cos_theta: f32,
    bb: Option<AxisBB>,
}

impl RotateX {
    pub fn new(p: Arc<HittableSS>, angle: f32) -> RotateX {
        let sin_theta = angle.to_radians().sin();
        let cos_theta = angle.to_radians().cos();

        let bbox = p.bounding_box(0.0, 1.0).unwrap();

        let mut min = Vec3::new_const(f32::INFINITY);
        let mut max = Vec3::new_const(f32::NEG_INFINITY);

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let x = if i == 1 { bbox.max.x } else { bbox.min.x };
                    let y = if j == 1 { bbox.max.y } else { bbox.min.y };
                    let z = if k == 1 { bbox.max.z } else { bbox.min.z };

                    let new_y = cos_theta * y - sin_theta * z;
                    let new_z = sin_theta * y + cos_theta * z;
                    let tester = Vec3::new(x, new_y, new_z);
                    for c in 0..3 {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        RotateX {
            ptr: p,
            sin_theta,
            cos_theta,
            bb: Some(AxisBB::new(min, max)),
        }
    }
}

impl Hittable for RotateX {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let mut origin = r.origin;
        let mut direction = r.direction;

        origin.y = self.cos_theta * r.origin.y + self.sin_theta * r.origin.z;
        origin.z = -self.sin_theta * r.origin.y + self.cos_theta * r.origin.z;

        direction.y = self.cos_theta * r.direction.y + self.sin_theta * r.direction.z;
        direction.z = -self.sin_theta * r.direction.y + self.cos_theta * r.direction.z;

        let rotated_r = Ray::new_with_time(origin, direction, r.time);

        if let Some(rec) = self.ptr.hit(&rotated_r, tmin, tmax) {
            let mut p = rec.p;
            let mut normal = rec.normal;

            p.y = self.cos_theta * rec.p.y - self.sin_theta * rec.p.z;
            p.z = self.sin_theta * rec.p.y + self.cos_theta * rec.p.z;

            normal.y = self.cos_theta * rec.normal.y - self.sin_theta * rec.normal.z;
            normal.z = self.sin_theta * rec.normal.y + self.cos_theta * rec.normal.z;

            let mut out_rec = HitRec::new(
                p,
                normal,
                rec.t,
                rec.u,
                rec.v,
                rec.front,
                rec.material.clone(),
            );
            out_rec.set_face_normal(&rotated_r, normal);

            return Some(out_rec);
        }

        None
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        self.bb
    }
}

pub struct RotateZ {
    ptr: Arc<HittableSS>,
    sin_theta: f32,
    cos_theta: f32,
    bb: Option<AxisBB>,
}

impl RotateZ {
    pub fn new(p: Arc<HittableSS>, angle: f32) -> RotateZ {
        let sin_theta = angle.to_radians().sin();
        let cos_theta = angle.to_radians().cos();

        let bbox = p.bounding_box(0.0, 1.0).unwrap();

        let mut min = Vec3::new_const(f32::INFINITY);
        let mut max = Vec3::new_const(f32::NEG_INFINITY);

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let x = if i == 1 { bbox.max.x } else { bbox.min.x };
                    let y = if j == 1 { bbox.max.y } else { bbox.min.y };
                    let z = if k == 1 { bbox.max.z } else { bbox.min.z };

                    let new_x = cos_theta * x - sin_theta * y;
                    let new_y = sin_theta * x + cos_theta * y;
                    let tester = Vec3::new(new_x, new_y, z);
                    for c in 0..3 {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        RotateZ {
            ptr: p,
            sin_theta,
            cos_theta,
            bb: Some(AxisBB::new(min, max)),
        }
    }
}

impl Hittable for RotateZ {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let mut origin = r.origin;
        let mut direction = r.direction;

        origin.x = self.cos_theta * r.origin.x + self.sin_theta * r.origin.y;
        origin.y = -self.sin_theta * r.origin.x + self.cos_theta * r.origin.y;

        direction.x = self.cos_theta * r.direction.x + self.sin_theta * r.direction.y;
        direction.y = -self.sin_theta * r.direction.x + self.cos_theta * r.direction.y;

        let rotated_r = Ray::new_with_time(origin, direction, r.time);

        if let Some(rec) = self.ptr.hit(&rotated_r, tmin, tmax) {
            let mut p = rec.p;
            let mut normal = rec.normal;

            p.x = self.cos_theta * rec.p.x - self.sin_theta * rec.p.y;
            p.y = self.sin_theta * rec.p.x + self.cos_theta * rec.p.y;

            normal.x = self.cos_theta * rec.normal.x - self.sin_theta * rec.normal.y;
            normal.y = self.sin_theta * rec.normal.x + self.cos_theta * rec.normal.y;

            let mut out_rec = HitRec::new(
                p,
                normal,
                rec.t,
                rec.u,
                rec.v,
                rec.front,
                rec.material.clone(),
            );
            out_rec.set_face_normal(&rotated_r, normal);

            return Some(out_rec);
        }

        None
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        self.bb
    }
}
