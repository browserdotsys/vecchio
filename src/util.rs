use crate::rand::Rng;
use crate::vec3::Vec3;

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
