#![allow(dead_code)]

#[macro_use]
extern crate derive_new;
extern crate rand;
extern crate rayon;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use rand::Rng;
use vec3::Vec3;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

mod vec3;

const ASPECT_RATIO : f32 = 16.0/9.0;
const WIDTH: usize = 1024;
const HEIGHT: usize = ((WIDTH as f32) / ASPECT_RATIO) as usize;
const SAMPLES_PER_PIXEL: usize = 100;
const MAX_DEPTH: u32 = 100;

#[derive(Copy,Clone)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
    time: f32,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray::new_with_time(origin, direction, 0.0)
    }

    fn new_with_time(origin: Vec3, direction: Vec3, time: f32) -> Ray {
        Ray { origin, direction, time }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(new)]
struct HitRec {
    p: Vec3,
    normal: Vec3,
    t: f32,
    u: f32,
    v: f32,
    front: bool,
    material: Arc<MaterialSS>,
}

impl HitRec {
    fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front = r.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front { outward_normal } else { -outward_normal };
    }
}

#[derive(new)]
struct Lambertian {
    albedo: Arc<TextureSS>,
}

#[derive(new)]
struct Dielectric {
    ref_idx: f32,
}

#[derive(new)]
struct Metal {
    albedo: Arc<TextureSS>,
    fuzz: f32,
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n*v.dot(n)*2.0
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = -uv.dot(n);
    let r_out_parallel = (uv + n * cos_theta) * etai_over_etat;
    let r_out_perp = n * -(1.0 - r_out_parallel.length2()).sqrt();
    r_out_parallel + r_out_perp
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0-ref_idx) / (1.0+ref_idx);
    r0 = r0*r0;
    r0 + (1.0-r0) * (1.0 - cosine).powf(5.0)
}

trait Material {
    fn scatter(&self, r: Ray, rec: &HitRec) -> (bool, Vec3, Ray) ;
}
type MaterialSS = dyn Material+Send+Sync;

impl Lambertian {
    fn random() -> Vec3 {
        let mut rng = rand::thread_rng();
        let a = rng.gen_range(0.0, 2.0*std::f32::consts::PI);
        let z = rng.gen_range(-1.0, 1.0);
        let r = ((1.0 - z*z) as f32).sqrt();

        Vec3::new(r*a.cos(), r*a.sin(), z)
    }
}

impl Material for Lambertian {
    fn scatter(&self, r: Ray, rec: &HitRec) -> (bool, Vec3, Ray) {
        let scatter_direction = rec.normal + Lambertian::random();
        let scattered = Ray::new_with_time(rec.p, scatter_direction, r.time);
        let attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        (true, attenuation, scattered)
    }
}

impl Material for Metal {
    fn scatter(&self, r: Ray, rec: &HitRec) -> (bool, Vec3, Ray) {
        let reflected = reflect(r.direction.unit_vector(), rec.normal);
        let scattered = Ray::new_with_time(rec.p, reflected + random_in_unit_sphere()*self.fuzz, r.time);
        let attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        let did_scatter = scattered.direction.dot(rec.normal) > 0.0;
        (did_scatter, attenuation, scattered)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r: Ray, rec: &HitRec) -> (bool, Vec3, Ray) {
        let mut rng = rand::thread_rng();
        let attenuation = Vec3::new_const(1.0);
        let etai_over_etat = if rec.front { 1.0 / self.ref_idx } else { self.ref_idx };
        let unit_direction = r.direction.unit_vector();
        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
        if etai_over_etat * sin_theta > 1.0 {
            let reflected = reflect(unit_direction, rec.normal);
            let scattered = Ray::new_with_time(rec.p, reflected, r.time);
            return (true, attenuation, scattered);
        }
        let reflect_prob = schlick(cos_theta, etai_over_etat);
        if rng.gen::<f32>() < reflect_prob {
            let reflected = reflect(unit_direction, rec.normal);
            let scattered = Ray::new_with_time(rec.p, reflected, r.time);
            return (true, attenuation, scattered);
        }
        let refracted = refract(unit_direction, rec.normal, etai_over_etat);
        let scattered = Ray::new_with_time(rec.p, refracted, r.time);
        (true, attenuation, scattered)
    }
}


fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::random_range(-1.0,1.0);
        if p.length2() >= 1.0 { continue; }
        return p;
    }
}

fn random_in_unit_disk() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0, 1.0),
            rng.gen_range(-1.0, 1.0),
            0.0
        );
        if p.length2() >= 1.0 { continue; }
        return p;
    }
}

trait Texture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 ;
}
type TextureSS = dyn Texture+Send+Sync;


#[derive(new)]
struct SolidColor {
    color_value: Vec3,
}

impl Texture for SolidColor {
    fn value(&self, _u: f32, _v: f32, _p: Vec3) -> Vec3 {
        self.color_value
    }
}

#[derive(new)]
struct Checker {
    odd: Arc<TextureSS>,
    even: Arc<TextureSS>,
}

impl Texture for Checker {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        let sins = (10.0*p.x).sin()*(10.0*p.y).sin()*(10.0*p.z).sin();
        if sins < 0.0 {
            return self.odd.value(u,v,p);
        }
        else {
            return self.even.value(u,v,p);
        }
    }
}

struct ImageTexture {
    buf: Vec<u8>,
    width: usize,
    height: usize,
}

const BPP: usize = 3;
impl ImageTexture {
    fn new(path: &str) -> ImageTexture {
        let decoder = png::Decoder::new(File::open(path).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut buf).unwrap();
        ImageTexture {
            buf: buf,
            width: info.width as usize,
            height: info.height as usize,
        }
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f32, v: f32, _p: Vec3) -> Vec3 {
        let u = Vec3::clamp(u, 0.0, 1.0);
        let v = 1.0 - Vec3::clamp(v, 0.0, 1.0);
        let mut i = (u * self.width as f32) as usize;
        let mut j = (v * self.height as f32) as usize;
        if i >= self.width { i = self.width - 1; }
        if j >= self.height { j = self.height - 1; }

        let buf_start = j*self.width*BPP + i*BPP;
        let pix = &self.buf[buf_start..buf_start+BPP];
        let color_scale = 1.0 / 255.0;
        Vec3::new(
            color_scale * pix[0] as f32,
            color_scale * pix[1] as f32,
            color_scale * pix[2] as f32,
        )
    }
}

trait Hittable {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> ;
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> ;
}

type HittableSS = dyn Hittable+Send+Sync;

#[derive(new)]
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Arc<MaterialSS>,
}

impl Sphere {
    fn to_spherical(p: Vec3) -> (f32, f32) {
        let pi = std::f32::consts::PI;
        let phi = p.z.atan2(p.x);
        let theta = p.y.asin();
        let u = 1.0-((phi + pi)/(2.0*pi));
        let v = (theta + pi/2.0) / pi;
        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let oc = r.origin - self.center;
        let a = r.direction.length2();
        let half_b = oc.dot(r.direction);
        let c = oc.length2() - self.radius*self.radius;
        let discriminant = half_b*half_b - a*c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            for temp in &[(-half_b - root) / a, (-half_b + root) / a] {
                if tmin < *temp && *temp < tmax {
                    let mut ret = HitRec::new(
                            r.at(*temp),
                            (r.at(*temp) - self.center) / self.radius,
                            *temp,
                            0.0, 0.0,
                            false,
                            self.material.clone(),
                    );
                    ret.set_face_normal(r, (ret.p - self.center) / self.radius);
                    let (u, v) = Sphere::to_spherical((ret.p - self.center) / self.radius);
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
struct MovingSphere {
    center0: Vec3,
    center1: Vec3,
    time0: f32,
    time1: f32,
    radius: f32,
    material: Arc<MaterialSS>,
}

impl MovingSphere {
    fn center(&self, time: f32) -> Vec3 {
        self.center0 + (self.center1 - self.center0) *
                       ((time - self.time0) / (self.time1 - self.time0))
    }
}

impl Hittable for MovingSphere {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let oc = r.origin - self.center(r.time);
        let a = r.direction.length2();
        let half_b = oc.dot(r.direction);
        let c = oc.length2() - self.radius*self.radius;
        let discriminant = half_b*half_b - a*c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            for temp in &[(-half_b - root) / a, (-half_b + root) / a] {
                if tmin < *temp && *temp < tmax {
                    let mut ret = HitRec::new(
                            r.at(*temp),
                            (r.at(*temp) - self.center(r.time)) / self.radius,
                            *temp,
                            0.0, 0.0,
                            false,
                            self.material.clone()
                    );
                    ret.set_face_normal(r, (ret.p - self.center(r.time)) / self.radius);
                    let (u, v) = Sphere::to_spherical((ret.p - self.center(r.time)) / self.radius);
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
struct Rect {
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
    fn XYRect(x0: f32, x1: f32, y0: f32, y1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(x0, x1, y0, y1, k, 0, 1, 2, mat)
    }

    #[allow(non_snake_case)]
    fn XZRect(x0: f32, x1: f32, z0: f32, z1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(x0, x1, z0, z1, k, 0, 2, 1, mat)
    }

    #[allow(non_snake_case)]
    fn YZRect(y0: f32, y1: f32, z0: f32, z1: f32, k: f32, mat: Arc<MaterialSS>) -> Rect {
        Rect::new(y0, y1, z0, z1, k, 1, 2, 0, mat)
    }
}

impl Hittable for Rect {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        let t = (self.k-r.origin[self.axis2]) / r.direction[self.axis2];
        if t < tmin || t > tmax {
            return None;
        }
        let a = r.origin[self.axis0] + t*r.direction[self.axis0];
        let b = r.origin[self.axis1] + t*r.direction[self.axis1];
        if a < self.c0 || a > self.c1 || b < self.d0 || b > self.d1 {
            return None;
        }
        let mut outward_normal = Vec3::new_const(0.0);
        outward_normal[self.axis2] = 1.0;
        let u = (a-self.c0)/(self.c1-self.c0);
        let v = (b-self.d0)/(self.d1-self.d0);

        let mut ret = HitRec::new(
            r.at(t),
            Vec3::new_const(0.0), // overwritten later
            t, u, v,
            false,                // overwritten later
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
struct FlipFace {
    ptr: Arc<HittableSS>
}

impl Hittable for FlipFace {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        if let Some(h) = self.ptr.hit(r, tmin, tmax) {
            Some(HitRec::new(h.p, h.normal, h.t, h.u, h.v, !h.front, h.material))
        }
        else {
            return None;
        }
    }
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        self.ptr.bounding_box(t0, t1)
    }
}

struct Boxy {
    box_min: Vec3,
    box_max: Vec3,
    sides: Vec<Arc<HittableSS>>,
}

impl Boxy {
    fn new(p0: Vec3, p1: Vec3, mat: Arc<MaterialSS>) -> Boxy {
        assert!(p0.x < p1.x);
        assert!(p0.y < p1.y);
        assert!(p0.z < p1.z);
        let sides : Vec<Arc<HittableSS>> = vec![
            Arc::new(Rect::XYRect(p0.x, p1.x, p0.y, p1.y, p1.z, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::XYRect(p0.x, p1.x, p0.y, p1.y, p0.z, mat.clone())))),
            Arc::new(Rect::XZRect(p0.x, p1.x, p0.z, p1.z, p1.y, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::XZRect(p0.x, p1.x, p0.z, p1.z, p0.y, mat.clone())))),
            Arc::new(Rect::YZRect(p0.y, p1.y, p0.z, p1.z, p1.x, mat.clone())),
            Arc::new(FlipFace::new(Arc::new(Rect::YZRect(p0.y, p1.y, p0.z, p1.z, p0.x, mat.clone())))),
        ];
        Boxy { box_min: p0, box_max: p1, sides }
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
        let mut closest_rec : Option<HitRec> = None;
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

        let mut output_box : Option<AxisBB> = None;
        let mut first_box = true;

        for o in self {
            if let Some(bb) = o.bounding_box(t0, t1) {
                if first_box {
                    output_box = Some(bb);
                }
                else {
                    output_box = Some(AxisBB::surrounding_box(bb, output_box.unwrap()));
                }
            }
            else {
                return None;
            }
            first_box = false;
        }

        output_box
    }
}

// Axis-aligned bounding box
#[derive(new,Copy,Clone)]
struct AxisBB {
    min: Vec3,
    max: Vec3,
}

fn fmin(f1: f32, f2: f32) -> f32 {
    f1.min(f2)
}

fn fmax(f1: f32, f2: f32) -> f32 {
    f1.max(f2)
}

impl AxisBB {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> bool {
        let mut tmin_local = tmin;
        let mut tmax_local = tmax;
        for a in 0..3 {
            let t0 = fmin((self.min[a] - r.origin[a]) / r.direction[a],
                          (self.max[a] - r.origin[a]) / r.direction[a]);
            let t1 = fmax((self.min[a] - r.origin[a]) / r.direction[a],
                          (self.max[a] - r.origin[a]) / r.direction[a]);
            tmin_local = fmax(t0, tmin_local);
            tmax_local = fmin(t1, tmax_local);
            if tmax_local <= tmin_local {
                return false;
            }
        }
        return true;
    }

    fn surrounding_box(box1: AxisBB, box2: AxisBB) -> AxisBB {
        let small = Vec3::new(
            box1.min.x.min(box2.min.x),
            box1.min.y.min(box2.min.y),
            box1.min.z.min(box2.min.z),
        );
        let big = Vec3::new(
            box1.max.x.max(box2.max.x),
            box1.max.y.max(box2.max.y),
            box1.max.z.max(box2.max.z),
        );
        AxisBB::new(small, big)
    }
}

struct BVHNode {
    left: Arc<HittableSS>,
    right: Arc<HittableSS>,
    bb: AxisBB,
}

impl Hittable for BVHNode {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        if !self.bb.hit(r, tmin, tmax) {
            return None;
        }

        let rec_left = self.left.hit(r, tmin, tmax);
        let tmax_new = if let Some(r) = rec_left.as_ref() { r.t } else { tmax };
        let rec_right = self.right.hit(r, tmin, tmax_new);
        match (rec_left.as_ref(), rec_right.as_ref()) {
            (Some(l), Some(r)) => {
                if l.t < r.t {
                    return rec_left;
                }
                else {
                    return rec_right;
                }
            },
            (Some(_), None) => {
                return rec_left;
            },
            (None, Some(_)) => {
                return rec_right;
            },
            (None, None) => {
                return None;
            }
        }
    }

    fn bounding_box(&self, _t0: f32, _t1: f32) -> Option<AxisBB> {
        Some(self.bb)
    }
}

impl BVHNode {
    fn box_for_hittables(h1: &Arc<HittableSS>, h2: &Arc<HittableSS>) -> AxisBB {
        AxisBB::surrounding_box(
            h1.bounding_box(0.0, 0.0).unwrap(),
            h2.bounding_box(0.0, 0.0).unwrap()
        )
    }

    fn new(objects: &mut [Arc<HittableSS>]) -> BVHNode {
        let mut rng = rand::thread_rng();
        let axis = rng.gen_range(0,3);

        if objects.len() == 1 {
            return BVHNode {
                left: objects[0].clone(),
                right: objects[0].clone(),
                bb: BVHNode::box_for_hittables(&objects[0], &objects[0]),
            };
        }
        else if objects.len() == 2 {
            let a_bb = objects[0].bounding_box(0.0, 0.0).unwrap();
            let b_bb = objects[1].bounding_box(0.0, 0.0).unwrap();
            let (mut i1, mut i2) = (0,1);
            if a_bb.min[axis] < b_bb.min[axis] {
                i1 = 1;
                i2 = 0;
            }
            return BVHNode {
                left: objects[i1].clone(),
                right: objects[i2].clone(),
                bb: BVHNode::box_for_hittables(&objects[i1], &objects[i2]),
            };
        }
        else {
            objects.sort_by(
                |a,b| {
                    let a_bb = a.bounding_box(0.0, 0.0).unwrap();
                    let b_bb = b.bounding_box(0.0, 0.0).unwrap();
                    a_bb.min[axis].partial_cmp(&b_bb.min[axis]).unwrap()
                }
            );
            let (left_part, right_part) = objects.split_at_mut(objects.len()/2);
            let left = Arc::new(BVHNode::new(left_part));
            let right = Arc::new(BVHNode::new(right_part));
            let bb = AxisBB::surrounding_box(
                left.bounding_box(0.0, 0.0).unwrap(),
                right.bounding_box(0.0, 0.0).unwrap(),
            );
            return BVHNode { left, right, bb };
        }
    }
}

#[derive(new)]
struct Translate {
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
        }
        else {
            None
        }
    }
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        if let Some(bb) = self.ptr.bounding_box(t0, t1) {
            Some(AxisBB::new(
                bb.min + self.offset,
                bb.max + self.offset,
            ))
        }
        else {
            None
        }
    }
}

struct Rotate {
    ptr: Arc<HittableSS>,
    sin_theta: f32,
    cos_theta: f32,
    bb: Option<AxisBB>,
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
    time0: f32,
    time1: f32,
}

impl Camera {
    fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3,
           vfov: f32, aspect_ratio: f32,
           aperture: f32, focus_dist: f32,
           time0: f32, time1: f32) -> Camera {
        let theta = vfov.to_radians();
        let h = (theta/2.0).tan();
        let viewport_height = h * 2.0;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).unit_vector();
        let u = vup.cross(w).unit_vector();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = u*viewport_width*focus_dist;
        let vertical = v*viewport_height*focus_dist;
        let lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - w*focus_dist;
        let lens_radius = aperture / 2.0;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u, v, w,
            lens_radius,
            time0, time1,
        }
    }

    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let mut rng = rand::thread_rng();
        let rd = random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new_with_time(self.origin + offset,
            self.lower_left_corner + self.horizontal*s + self.vertical*t - self.origin - offset,
            rng.gen_range(self.time0, self.time1),
        )
    }
}

fn ray_color(r: Ray, world: Arc<dyn Hittable>, depth: u32) -> Vec3 {
    if depth > MAX_DEPTH {
        return Vec3::new_const(0.0);
    }

    if let Some(c) = world.hit(&r, 0.001, f32::INFINITY) {
        let (did_scatter, attenuation, scattered) = c.material.scatter(r, &c);
        if did_scatter {
            return attenuation * ray_color(scattered, world, depth + 1);
        }
        return Vec3::new_const(0.0);
    }

    // No hit, draw background
    let u = r.direction.unit_vector();
    // t is the y position, scaled to viewport
    let t = 0.5 * (1.0 + u.y);
    // shade with a mix of white and blue according to position
    Vec3::new(0.5,0.7,1.0)*t + Vec3::new_const(1.0)*(1.0-t)
}

fn balls_demo(world: &mut Vec<Arc<HittableSS>>) {
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0,0.0,-1.0),
            0.5,
            Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.1, 0.2, 0.5))))),
        ),
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0,-100.5,-1.0),
            100.0,
            Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.8, 0.8, 0.0))))),
        ),
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(1.0,0.0,-1.0),
            0.5,
            Arc::new(Metal::new(
                    Arc::new(SolidColor::new(Vec3::new(0.8, 0.6, 0.2))),
                    0.3
            )),
        ),
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(-1.0,0.0,-1.0),
            0.5,
            Arc::new(Dielectric::new(1.5)),
        ),
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(-1.0,0.0,-1.0),
            -0.45,
            Arc::new(Dielectric::new(1.5)),
        ),
    ));
}

fn random_spheres_demo(world: &mut Vec<Arc<HittableSS>>) {
    // Ground
    let checker = Checker::new(
        Arc::new(SolidColor::new(Vec3::new(0.1, 0.1, 0.1))),
        Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))),
    );
    let ground_material = Arc::new(
        Lambertian::new(Arc::new(checker))
    );
    world.push(Arc::new(Sphere::new(Vec3::new(0.0,-1000.0,0.0), 1000.0, ground_material)));

    // Random spheres
    let mut rng = rand::thread_rng();
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen::<f32>();
            let center = Vec3::new(
                a as f32 + 0.9*rng.gen::<f32>(),
                0.2,
                b as f32 + 0.9*rng.gen::<f32>(),
            );

            if (center - Vec3::new(4.0,0.2,0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = Vec3::random() * Vec3::random();
                    world.push(Arc::new(
                        Sphere::new(
                            center, 0.2,
                            Arc::new(Lambertian::new(Arc::new(SolidColor::new(albedo))))
                        ))
                    );
                }
                else if choose_mat < 0.95 {
                    let albedo = SolidColor::new(Vec3::random_range(0.5, 1.0));
                    let fuzz = rng.gen_range(0.0, 0.5);
                     world.push(Arc::new(
                        Sphere::new(
                            center, 0.2,
                            Arc::new(Metal::new(Arc::new(albedo),fuzz))
                        ))
                    );
                }
                else {
                    world.push(Arc::new(
                        Sphere::new(
                            center, 0.2,
                            Arc::new(Dielectric::new(1.5))
                        ))
                    );
                }
            }
        }
    }

    // Three big spheres
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0, 1.0, 0.0), 1.0,
            Arc::new(Dielectric::new(1.5))
        )
    ));
    /*
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(-4.0, 1.0, 0.0), 1.0,
            Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.4, 0.2, 0.1)))))
        ))
    );
    */
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(-4.0, 1.0, 0.0), 1.0,
            Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/earthmap.png"))))
        )
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(4.0, 1.0, 0.0), 1.0,
            Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.7, 0.6, 0.5))), 0.0))
        )
    ));

    // Mirrors
    world.push(Arc::new(
        Rect::XYRect(-8.0, 0.0, 0.0, 3.5, -3.0,
            Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))), 0.0))
        )
    ));
    world.push(Arc::new(
        Rect::XYRect(-8.0, -0.0, 0.0, 3.5, 3.0,
            Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))), 0.0))
        )
    ));
}

fn two_spheres_demo(world: &mut Vec<Arc<HittableSS>>) {
    let checker = Checker::new(
        Arc::new(SolidColor::new(Vec3::new(0.2, 0.3, 0.1))),
        Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))),
    );
    let ground_material = Arc::new(
        Lambertian::new(Arc::new(checker))
    );
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0, -10.0, 0.0), 10.0,
            ground_material.clone()
        )
    ));
    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0, 10.0, 0.0), 10.0,
            ground_material.clone()
        )
    ));
}

struct Bowser {
    parts: Arc<HittableSS>,
}

impl Bowser {
    fn new(x: f32, y: f32, z: f32) -> Bowser {
        let mut world : Vec<Arc<HittableSS>> = vec![];
        // Face
        world.push(Arc::new(
            Rect::XYRect(x-2.0, x+2.0, y+1.0, y+4.0, z-3.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_face.png"))))
            )
        ));
        // Top
        world.push(Arc::new(
            Rect::XZRect(x-2.0, x+2.0, z-6.0, z-3.0, y+4.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_top.png"))))
            )
        ));
        // Back
        world.push(Arc::new(
            Rect::XYRect(x-2.0, x+2.0, y+1.0, y+4.0, z-6.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_back.png"))))
            )
        ));
        // Sides
        world.push(Arc::new(
            Rect::YZRect(y+1.0, y+4.0, z-6.0, z-3.0, x-2.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_side.png"))))
            )
        ));
        world.push(Arc::new(
            Rect::YZRect(y+1.0, y+4.0, z-6.0, z-3.0, x+2.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_side.png"))))
            )
        ));
        // Bottom
        let grey = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.278, 0.387, 0.438)))));
        world.push(Arc::new(
            Rect::XZRect(x-2.0, x+2.0, z-6.0, z-3.0, y+1.0, grey.clone()
            )
        ));
        // Feet
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.5, y+0.5, z-4.75),
                Vec3::new(x-0.5, y+1.0, z-4.25),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.5, y+0.5, z-4.75),
                Vec3::new(x+1.5, y+1.0, z-4.25),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.5, y+0.25, z-4.75),
                Vec3::new(x-0.5, y+0.5, z-3.5),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.5, y+0.25, z-4.75),
                Vec3::new(x+1.5, y+0.5, z-3.5),
                grey.clone(),
            )
        ));
        // Arms
        let brown = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.4, 0.2, 0.1)))));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.25, y+1.75, z-4.65),
                Vec3::new(x-2.00, y+2.75, z-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.50, y+1.75, z-4.65),
                Vec3::new(x-2.25, y+2.50, z-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.75, y+1.75, z-4.65),
                Vec3::new(x-2.50, y+2.25, z-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.00, y+1.75, z-4.65),
                Vec3::new(x+2.25, y+2.75, z-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.25, y+1.75, z-4.65),
                Vec3::new(x+2.50, y+2.50, z-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.50, y+1.75, z-4.65),
                Vec3::new(x+2.75, y+2.25, z-4.35),
                brown.clone(),
            )
        ));
        // Face rim
        let lightgrey = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.601, 0.687, 0.723)))));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y+3.875, z-3.00),
                Vec3::new(x+2.0, y+4.00, z-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y+1.0, z-3.00),
                Vec3::new(x+2.0, y+1.125, z-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y+1.125, z-3.00),
                Vec3::new(x-1.875, y+3.875, z-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+1.875, y+1.125, z-3.00),
                Vec3::new(x+2.0, y+3.875, z-2.875),
                lightgrey.clone(),
            )
        ));

        // Butt ports
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y+1.625, z-6.125),
                Vec3::new(x-0.875, y+1.75, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y+1.125, z-6.125),
                Vec3::new(x-0.875, y+1.25, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y+1.25, z-6.125),
                Vec3::new(x-1.750, y+1.625, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.0, y+1.25, z-6.125),
                Vec3::new(x-0.875, y+1.625, z-6.0),
                lightgrey.clone(),
            )
        ));

        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y+1.625, z-6.125),
                Vec3::new(x+1.875, y+1.75, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y+1.125, z-6.125),
                Vec3::new(x+1.875, y+1.25, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+1.750, y+1.25, z-6.125),
                Vec3::new(x+1.875, y+1.625, z-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y+1.25, z-6.125),
                Vec3::new(x+1.0, y+1.625, z-6.0),
                lightgrey.clone(),
            )
        ));

        // Mirrors
        world.push(Arc::new(
            Rect::XYRect(-5.0, 5.0, 0.0, 6.0, -8.0,
                Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))), 0.0))
            )
        ));
        world.push(Arc::new(
            Rect::XYRect(-5.0, 5.0, 0.0, 6.0, 8.0,
                Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))), 0.0))
            )
        ));

        // Mirror frame
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(-5.25, 6.0, 7.75),
                Vec3::new(5.25, 6.25, 8.0),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(-5.25, 0.0, 7.75),
                Vec3::new(-5.00, 6.0, 8.0),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(5.00, 0.0, 7.75),
                Vec3::new(5.25, 6.0, 8.0),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(-5.25, 6.0, -8.0),
                Vec3::new(5.25, 6.25, -7.75),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(-5.25, 0.0, -8.0),
                Vec3::new(-5.00, 6.0, -7.75),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(5.00, 0.0, -8.8),
                Vec3::new(5.25, 6.0, -7.75),
                brown.clone(),
            )
        ));

        Bowser { parts: Arc::new(BVHNode::new(&mut world[..])) }
    }
}

impl Hittable for Bowser {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> {
        self.parts.hit(r, tmin, tmax)
    }
    fn bounding_box(&self, t0: f32, t1: f32) -> Option<AxisBB> {
        self.parts.bounding_box(t0, t1)
    }
}

fn bowser_demo(world: &mut Vec<Arc<HittableSS>>) {
    // Ground
    let checker = Checker::new(
        Arc::new(SolidColor::new(Vec3::new(0.1, 0.1, 0.1))),
        Arc::new(SolidColor::new(Vec3::new(0.9, 0.9, 0.9))),
    );
    let ground_material = Arc::new(
        Lambertian::new(Arc::new(checker))
    );
    world.push(Arc::new(Sphere::new(Vec3::new(0.0,-1000.0,0.0), 1000.0, ground_material)));

    // Bowser! Goofed a little when creating by putting him .25 units
    // off the ground, so fix that with a Translate here
    world.push(Arc::new(Translate::new(
        Arc::new(Bowser::new(0.0,0.0,0.0)),
        Vec3::new(0.0, -0.25, 0.0),
    )));
}

fn main() -> Result<(), std::io::Error> {
    let mut pixels: Vec<Vec3> = vec![Vec3::new_const(0.0); WIDTH * HEIGHT];

    // Camera and world
    // let (cam, world) = balls_demo();
    eprintln!("Generating scene...");
    let mut world_vec : Vec<Arc<HittableSS>> = vec![];
    //random_spheres_demo(&mut world_vec);
    //balls_demo(&mut world_vec);
    //two_spheres_demo(&mut world_vec);
    bowser_demo(&mut world_vec);
    let world_bvh = Arc::new(BVHNode::new(&mut world_vec[..]));

    let radius = 20.0;

    let mut angle = 84.5_f32;
    let mut file_idx = 0;
    while angle < 360.0 {
        // Timing
        let start = Instant::now();

        // Set up camera
        let look_x = radius*(angle).to_radians().cos();
        let look_z = radius*(angle).to_radians().sin();
        //let lookfrom = Vec3::new(look_x,1.5,look_z);
        let lookfrom = Vec3::new(look_x,2.5,-3.0);
        let lookat = Vec3::new(0.0,1.5,10.0);
        let vup = Vec3::new(0.0,1.0,0.0);
        let dist_to_focus = 10.0;
        let aperture = 0.0;

        let cam = Camera::new(lookfrom, lookat, vup, 20.0, ASPECT_RATIO, aperture, dist_to_focus, 0.0, 1.0);

        // Trace rays
        pixels.par_iter_mut().enumerate().for_each(|(i,pix)| {
            let x = i % WIDTH;
            let y = i / WIDTH;
            let mut rng = rand::thread_rng();
            let mut c = Vec3::new_const(0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = ((x as f32)+rng.gen::<f32>()) / ((WIDTH-1) as f32);
                let v = ((y as f32)+rng.gen::<f32>()) / ((HEIGHT-1) as f32);
                let ray = cam.get_ray(u,v);
                let color = ray_color(ray, world_bvh.clone(), 1);
                c += color;
            }
            c /= SAMPLES_PER_PIXEL as f32;
            *pix = c;
        });

        // Write output
        let filename = format!("output_{:04}.ppm", file_idx);
        let file = File::create(filename.clone()).unwrap();
        let mut wr = BufWriter::new(&file);

        writeln!(&mut wr, "P3")?;
        writeln!(&mut wr, "{} {}", WIDTH, HEIGHT)?;
        writeln!(&mut wr, "255")?;

        for y in {0..HEIGHT}.rev() {
            for x in 0..WIDTH {
                let (ir, ig, ib) = pixels[y*WIDTH+x].to_color();
                writeln!(&mut wr, "{} {} {}", ir, ig, ib)?;
            }
        }
        eprintln!("Wrote frame {} in {}s", filename, start.elapsed().as_secs());

        angle += 0.5;
        file_idx += 1;
    }

    Ok(())
}
