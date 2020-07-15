use crate::hittable::HittableSS;
use crate::rand::Rng;
use crate::vec3::Vec3;
use std::sync::Arc;

pub fn fmin(f1: f32, f2: f32) -> f32 {
    f1.min(f2)
}

pub fn fmax(f1: f32, f2: f32) -> f32 {
    f1.max(f2)
}

pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n * v.dot(n) * 2.0
}

pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = -uv.dot(n);
    let r_out_parallel = (uv + n * cos_theta) * etai_over_etat;
    let r_out_perp = n * -(1.0 - r_out_parallel.length2()).sqrt();
    r_out_parallel + r_out_perp
}

pub fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

pub fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::random_range(-1.0, 1.0);
        if p.length2() >= 1.0 {
            continue;
        }
        return p;
    }
}

pub fn random_in_unit_disk() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let p = Vec3::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0);
        if p.length2() >= 1.0 {
            continue;
        }
        return p;
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

pub struct ONB {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

impl std::ops::Index<usize> for ONB {
    type Output = Vec3;
    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.u,
            1 => &self.v,
            2 => &self.w,
            _ => unreachable!(),
        }
    }
}

impl std::ops::IndexMut<usize> for ONB {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.u,
            1 => &mut self.v,
            2 => &mut self.w,
            _ => unreachable!(),
        }
    }
}

impl ONB {
    pub fn local(&self, a: Vec3) -> Vec3 {
        self.u*a.x + self.v*a.y + self.w*a.z
    }

    pub fn new_from_w(n: Vec3) -> ONB {
        let w = n.unit_vector();
        let a = if w.x.abs() > 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        }
        else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let v = w.cross(a).unit_vector();
        let u = w.cross(v);
        ONB { u, v, w }
    }
}

pub trait PDF {
    fn value(&self, direction: Vec3) -> f32;

    fn generate(&self) -> Vec3;
}

pub type PDFSS = dyn PDF + Send + Sync;

pub struct CosinePDF {
    uvw: ONB,
}

impl CosinePDF {
    pub fn new(w: Vec3) -> CosinePDF {
        CosinePDF {
            uvw: ONB::new_from_w(w)
        }
    }
}

impl PDF for CosinePDF {
    fn value(&self, direction: Vec3) -> f32 {
        let cos = direction.unit_vector().dot(self.uvw.w);
        if cos <= 0.0 {
            0.0
        }
        else {
            cos / std::f32::consts::PI
        }
    }

    fn generate(&self) -> Vec3 {
        self.uvw.local(random_cosine_direction())
    }
}

#[derive(new)]
pub struct HittablePDF {
    ptr: Arc<HittableSS>,
    o: Vec3,
}

impl PDF for HittablePDF {
    fn value(&self, direction: Vec3) -> f32 {
        self.ptr.pdf_value(self.o, direction)
    }
    fn generate(&self) -> Vec3 {
        self.ptr.random(self.o)
    }
}

#[derive(new)]
pub struct MixturePDF {
    ptr1: Arc<PDFSS>,
    f1: f32,
    ptr2: Arc<PDFSS>,
    f2: f32,
}

impl PDF for MixturePDF {
    fn value(&self, direction: Vec3) -> f32 {
        self.f1 * self.ptr1.value(direction) + self.f2 * self.ptr2.value(direction)
    }

    fn generate(&self) -> Vec3 {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.f1 {
            self.ptr1.generate()
        }
        else {
            self.ptr2.generate()
        }
    }
}
