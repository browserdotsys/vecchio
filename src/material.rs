// Materials and textures

use crate::hittable::HitRec;
use crate::rand::Rng;
use crate::util::{random_in_unit_sphere, reflect, refract, schlick, CosinePDF, PDFSS};
use crate::vec3::Vec3;
use crate::Ray;
use array_init::array_init;
use rand::seq::SliceRandom;
use std::fs::File;
use std::sync::Arc;

#[derive(new)]
pub struct ScatterRec {
    pub specular_ray: Option<Ray>,
    pub attenuation: Vec3,
    pub pdf: Arc<PDFSS>,
}

pub trait Material {
    fn scatter(&self, r: Ray, rec: &HitRec) -> Option<(Vec3, Ray)> {
        if let Some(srec) = self.scatter_with_pdf(r, rec) {
            Some((srec.attenuation, srec.specular_ray.unwrap()))
        }
        else {
            None
        }
    }

    fn scatter_with_pdf(&self, _r: Ray, _rec: &HitRec) -> Option<ScatterRec> {
        None
    }

    fn scattering_pdf(&self, _r: Ray, _rec: &HitRec, _s: Ray) -> f32 {
        0.0
    }

    fn emitted(&self, _rec: &HitRec, _u: f32, _v: f32, _p: Vec3) -> Vec3 {
        Vec3::new_const(0.0)
    }
}

pub type MaterialSS = dyn Material + Send + Sync;

#[derive(new)]
pub struct Lambertian {
    albedo: Arc<TextureSS>,
}

impl Lambertian {
    fn random() -> Vec3 {
        let mut rng = rand::thread_rng();
        let a = rng.gen_range(0.0, 2.0 * std::f32::consts::PI);
        let z = rng.gen_range(-1.0, 1.0);
        let r = ((1.0 - z * z) as f32).sqrt();

        Vec3::new(r * a.cos(), r * a.sin(), z)
    }

    fn random_in_hemisphere(normal: Vec3) -> Vec3 {
        let in_unit_sphere = random_in_unit_sphere();
        if in_unit_sphere.dot(normal) > 0.0 {
            in_unit_sphere
        }
        else {
            -in_unit_sphere
        }
    }

    fn random_cosine_direction() -> Vec3 {
        let mut rng = rand::thread_rng();
        let r1 = rng.gen::<f32>();
        let r2 = rng.gen::<f32>();
        let z = (1.0-r2).sqrt();

        let phi = 2.0*r1*std::f32::consts::PI;
        let x = phi.cos()*r2.sqrt();
        let y = phi.sin()*r2.sqrt();

        Vec3::new(x, y, z)
    }
}

impl Material for Lambertian {
    fn scatter(&self, r: Ray, rec: &HitRec) -> Option<(Vec3,Ray)> {
        let scatter_direction = rec.normal + Lambertian::random();
        let scattered = Ray::new_with_time(rec.p, scatter_direction, r.time);
        let attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        Some((attenuation, scattered))
    }

    fn scatter_with_pdf(&self, _r: Ray, rec: &HitRec) -> Option<ScatterRec> {
        Some(ScatterRec::new(
            None,
            self.albedo.value(rec.u, rec.v, rec.p),
            Arc::new(CosinePDF::new(rec.normal)),
        ))
    }

    fn scattering_pdf(&self, _r: Ray, rec: &HitRec, s: Ray) -> f32 {
        let cos = rec.normal.dot(s.direction.unit_vector());
        if cos < 0.0 {
            0.0
        }
        else {
            cos / std::f32::consts::PI
        }
    }
}

#[derive(new)]
pub struct Metal {
    albedo: Arc<TextureSS>,
    fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, r: Ray, rec: &HitRec) -> Option<(Vec3,Ray)> {
        let reflected = reflect(r.direction.unit_vector(), rec.normal);
        let scattered = Ray::new_with_time(
            rec.p,
            reflected + random_in_unit_sphere() * self.fuzz,
            r.time,
        );
        let attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        if scattered.direction.dot(rec.normal) > 0.0 {
            Some((attenuation, scattered))
        }
        else {
            None
        }
    }

    fn scatter_with_pdf(&self, r: Ray, rec: &HitRec) -> Option<ScatterRec> {
        let reflected = reflect(r.direction.unit_vector(), rec.normal);
        Some(ScatterRec::new(
            Some(Ray::new(rec.p, reflected+random_in_unit_sphere()*self.fuzz)),
            self.albedo.value(rec.u, rec.v, rec.p),
            Arc::new(CosinePDF::new(rec.normal)), // No null pointers in rust - make it an Option<Arc>?
        ))
    }
}

#[derive(new)]
pub struct Dielectric {
    ref_idx: f32,
}

impl Material for Dielectric {
    fn scatter(&self, r: Ray, rec: &HitRec) -> Option<(Vec3,Ray)> {
        let mut rng = rand::thread_rng();
        let attenuation = Vec3::new_const(1.0);
        let etai_over_etat = if rec.front {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };
        let unit_direction = r.direction.unit_vector();
        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if etai_over_etat * sin_theta > 1.0 {
            let reflected = reflect(unit_direction, rec.normal);
            let scattered = Ray::new_with_time(rec.p, reflected, r.time);
            return Some((attenuation, scattered));
        }
        let reflect_prob = schlick(cos_theta, etai_over_etat);
        if rng.gen::<f32>() < reflect_prob {
            let reflected = reflect(unit_direction, rec.normal);
            let scattered = Ray::new_with_time(rec.p, reflected, r.time);
            return Some((attenuation, scattered));
        }
        let refracted = refract(unit_direction, rec.normal, etai_over_etat);
        let scattered = Ray::new_with_time(rec.p, refracted, r.time);
        Some((attenuation, scattered))
    }

    fn scatter_with_pdf(&self, r: Ray, rec: &HitRec) -> Option<ScatterRec> {
        let mut rng = rand::thread_rng();
        let mut srec = ScatterRec::new(
            None,                                 // fill in later
            Vec3::new_const(1.0),                 // white
            Arc::new(CosinePDF::new(rec.normal)), // No null pointers in rust - make it an Option<Arc>?
        );
        let etai_over_etat = if rec.front {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };
        let unit_direction = r.direction.unit_vector();
        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if etai_over_etat * sin_theta > 1.0 {
            let reflected = reflect(unit_direction, rec.normal);
            srec.specular_ray = Some(Ray::new_with_time(rec.p, reflected, r.time));
            return Some(srec);
        }
        let reflect_prob = schlick(cos_theta, etai_over_etat);
        if rng.gen::<f32>() < reflect_prob {
            let reflected = reflect(unit_direction, rec.normal);
            srec.specular_ray = Some(Ray::new_with_time(rec.p, reflected, r.time));
            return Some(srec);
        }
        let refracted = refract(unit_direction, rec.normal, etai_over_etat);
        srec.specular_ray = Some(Ray::new_with_time(rec.p, refracted, r.time));
        Some(srec)
    }
}

#[derive(new)]
pub struct DiffuseLight {
    emit: Arc<TextureSS>,
}

impl Material for DiffuseLight {
    fn scatter(&self, _r: Ray, _rec: &HitRec) -> Option<(Vec3,Ray)> {
        None
    }
    fn emitted(&self, rec: &HitRec, u: f32, v: f32, p: Vec3) -> Vec3 {
        if rec.front {
            self.emit.value(u, v, p)
        }
        else {
            Vec3::new_const(0.0)
        }
    }
}

pub trait Texture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3;
}
pub type TextureSS = dyn Texture + Send + Sync;

#[derive(new)]
pub struct SolidColor {
    color_value: Vec3,
}

impl Texture for SolidColor {
    fn value(&self, _u: f32, _v: f32, _p: Vec3) -> Vec3 {
        self.color_value
    }
}

#[derive(new)]
pub struct Checker {
    odd: Arc<TextureSS>,
    even: Arc<TextureSS>,
}

impl Texture for Checker {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        let sins = (10.0 * p.x).sin() * (10.0 * p.y).sin() * (10.0 * p.z).sin();
        if sins < 0.0 {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}

pub struct ImageTexture {
    buf: Vec<u8>,
    width: usize,
    height: usize,
}

const BPP: usize = 3;
impl ImageTexture {
    pub fn new(path: &str) -> ImageTexture {
        let decoder = png::Decoder::new(File::open(path).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut buf).unwrap();
        ImageTexture {
            buf,
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
        if i >= self.width {
            i = self.width - 1;
        }
        if j >= self.height {
            j = self.height - 1;
        }

        let buf_start = j * self.width * BPP + i * BPP;
        let pix = &self.buf[buf_start..buf_start + BPP];
        let color_scale = 1.0 / 255.0;
        Vec3::new(
            color_scale * pix[0] as f32,
            color_scale * pix[1] as f32,
            color_scale * pix[2] as f32,
        )
    }
}

pub struct Perlin {
    random_data: [Vec3; 256],
    perm_x: [usize; 256],
    perm_y: [usize; 256],
    perm_z: [usize; 256],
}

pub fn trilinear_interp(c: &[[[f32; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
    let mut accum: f32 = 0.0;
    for (i, c1) in c.iter().enumerate() {
        for (j, c2) in c1.iter().enumerate() {
            for (k, c3) in c2.iter().enumerate() {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                accum += (fi * u + (1.0 - fi) * (1.0 - u))
                    * (fj * v + (1.0 - fj) * (1.0 - v))
                    * (fk * w + (1.0 - fk) * (1.0 - w))
                    * c3;
            }
        }
    }
    accum
}

pub fn perlin_interp(c: &[[[Vec3; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
    let mut accum: f32 = 0.0;
    let uu = u * u * (3.0 - 2.0 * u);
    let vv = v * v * (3.0 - 2.0 * v);
    let ww = w * w * (3.0 - 2.0 * w);

    for (i, c1) in c.iter().enumerate() {
        for (j, c2) in c1.iter().enumerate() {
            for (k, c3) in c2.iter().enumerate() {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                let weight_v = Vec3::new(u - fi, v - fj, w - fk);
                accum += (fi * uu + (1.0 - fi) * (1.0 - uu))
                    * (fj * vv + (1.0 - fj) * (1.0 - vv))
                    * (fk * ww + (1.0 - fk) * (1.0 - ww))
                    * c3.dot(weight_v);
            }
        }
    }
    accum
}

impl Perlin {
    const NUM_POINTS: usize = 256;

    pub fn new() -> Perlin {
        let mut rng = rand::thread_rng();

        let random_data: [Vec3; Perlin::NUM_POINTS] =
            array_init(|_| Vec3::random_range(-1.0, 1.0).unit_vector());

        // Permutation arrays
        let mut perm_x: [usize; Perlin::NUM_POINTS] = array_init(|i| i);
        let mut perm_y: [usize; Perlin::NUM_POINTS] = array_init(|i| i);
        let mut perm_z: [usize; Perlin::NUM_POINTS] = array_init(|i| i);
        perm_x.shuffle(&mut rng);
        perm_y.shuffle(&mut rng);
        perm_z.shuffle(&mut rng);

        Perlin {
            random_data,
            perm_x,
            perm_y,
            perm_z,
        }
    }

    pub fn turb(&self, p: Vec3, depth: usize) -> f32 {
        let mut accum = 0.0;
        let mut temp_p = p;
        let mut weight = 1.0_f32;
        for _ in 0..depth {
            accum += weight * self.noise(temp_p);
            weight *= 0.5;
            temp_p *= 2.0;
        }

        accum.abs()
    }

    pub fn noise(&self, p: Vec3) -> f32 {
        let u = p.x - p.x.floor();
        let v = p.y - p.y.floor();
        let w = p.z - p.z.floor();

        let mut c = [[[Vec3::new_const(0.0); 2]; 2]; 2];

        let i = p.x.floor() as usize;
        let j = p.y.floor() as usize;
        let k = p.z.floor() as usize;
        for (di, c1) in c.iter_mut().enumerate() {
            for (dj, c2) in c1.iter_mut().enumerate() {
                for (dk, c3) in c2.iter_mut().enumerate() {
                    *c3 = self.random_data[self.perm_x[(i + di) & 255]
                        ^ self.perm_y[(j + dj) & 255]
                        ^ self.perm_z[(k + dk) & 255]];
                }
            }
        }

        perlin_interp(&c, u, v, w)
    }
}

pub struct NoiseTexture {
    noise: Perlin,
    scale: f32,
}

impl NoiseTexture {
    pub fn new(scale: f32) -> NoiseTexture {
        NoiseTexture {
            noise: Perlin::new(),
            scale,
        }
    }
}

impl Texture for NoiseTexture {
    fn value(&self, _u: f32, _v: f32, p: Vec3) -> Vec3 {
        Vec3::new_const(1.0) * 0.5 * (1.0 + (self.scale * p.z + 10.0 * self.noise.turb(p, 7)).sin())
    }
}

#[derive(new)]
pub struct Isotropic {
    albedo: Arc<TextureSS>,
}

impl Material for Isotropic {
    fn scatter(&self, r: Ray, rec: &HitRec) -> Option<(Vec3,Ray)> {
        let scattered = Ray::new_with_time(rec.p, random_in_unit_sphere(), r.time);
        let attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        Some((attenuation, scattered))
    }

    fn scatter_with_pdf(&self, _r: Ray, rec: &HitRec) -> Option<ScatterRec> {
        Some(ScatterRec::new(
            None,
            self.albedo.value(rec.u, rec.v, rec.p),
            Arc::new(CosinePDF::new(rec.normal)),
        ))
    }

    fn scattering_pdf(&self, _r: Ray, rec: &HitRec, s: Ray) -> f32 {
        let cos = rec.normal.dot(s.direction.unit_vector());
        if cos < 0.0 {
            0.0
        }
        else {
            cos / std::f32::consts::PI
        }
    }
}

#[derive(new)]
pub struct SpecDiffuse {
    specular: Arc<MaterialSS>,
    diffuse: Arc<MaterialSS>,
    pct: f32,
}

impl Material for SpecDiffuse {
    fn scatter_with_pdf(&self, r: Ray, rec: &HitRec) -> Option<ScatterRec> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.pct {
            self.specular.scatter_with_pdf(r, rec)
        }
        else {
            self.diffuse.scatter_with_pdf(r, rec)
        }
    }

    fn scattering_pdf(&self, r: Ray, rec: &HitRec, s: Ray) -> f32 {
        self.diffuse.scattering_pdf(r, rec, s)
    }
}
