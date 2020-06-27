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

mod vec3;

const ASPECT_RATIO : f32 = 16.0/9.0;
const WIDTH: usize = 640;
const HEIGHT: usize = ((WIDTH as f32) / ASPECT_RATIO) as usize;
const SAMPLES_PER_PIXEL: usize = 100;
const MAX_DEPTH: u32 = 50;

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
    front: bool,
    material: Material,
}

impl HitRec {
    fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front = r.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front { outward_normal } else { -outward_normal };
    }
}

#[derive(new,Copy,Clone)]
struct Lambertian {
    albedo: Vec3,
}

#[derive(new,Copy,Clone)]
struct Dielectric {
    ref_idx: f32,
}

#[derive(new,Copy,Clone)]
struct Metal {
    albedo: Vec3,
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

#[derive(Copy,Clone)]
enum Material {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

impl Material {
    fn scatter(&self, r: Ray, rec: &HitRec) -> (bool, Vec3, Ray) {
        match self {
            Material::Lambertian(ref l) => {
                let scatter_direction = rec.normal + Lambertian::random();
                let scattered = Ray::new_with_time(rec.p, scatter_direction, r.time);
                let attenuation = l.albedo;
                (true, attenuation, scattered)
            },
            Material::Metal(ref m) => {
                let reflected = reflect(r.direction.unit_vector(), rec.normal);
                let scattered = Ray::new_with_time(rec.p, reflected + random_in_unit_sphere()*m.fuzz, r.time);
                let attenuation = m.albedo;
                let did_scatter = scattered.direction.dot(rec.normal) > 0.0;
                (did_scatter, attenuation, scattered)
            }
            Material::Dielectric(ref d) => {
                let mut rng = rand::thread_rng();
                let attenuation = Vec3::new_const(1.0);
                let etai_over_etat = if rec.front { 1.0 / d.ref_idx } else { d.ref_idx };
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
    }
}

impl Lambertian {
    fn random() -> Vec3 {
        let mut rng = rand::thread_rng();
        let a = rng.gen_range(0.0, 2.0*std::f32::consts::PI);
        let z = rng.gen_range(-1.0, 1.0);
        let r = ((1.0 - z*z) as f32).sqrt();

        Vec3::new(r*a.cos(), r*a.sin(), z)
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

trait Hittable {
    fn hit(&self, r: &Ray, tmin: f32, tmax: f32) -> Option<HitRec> ;
}

#[derive(new)]
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
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
                            false,
                            self.material
                    );
                    ret.set_face_normal(r, (ret.p - self.center) / self.radius);
                    return Some(ret);
                }
            }
        }

        None
    }
}

#[derive(new)]
struct MovingSphere {
    center0: Vec3,
    center1: Vec3,
    time0: f32,
    time1: f32,
    radius: f32,
    material: Material,
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
                            false,
                            self.material
                    );
                    ret.set_face_normal(r, (ret.p - self.center(r.time)) / self.radius);
                    return Some(ret);
                }
            }
        }

        None
    }
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

fn ray_color(r: Ray, world: &Vec<Box<dyn Hittable+Send+Sync>>, depth: u32) -> Vec3 {
    if depth > MAX_DEPTH {
        return Vec3::new_const(0.0);
    }

    let mut tnear = f32::INFINITY;
    let mut closest : Option<HitRec> = None;
    for w in world {
        if let Some(rec) = w.hit(&r, 0.001, tnear) {
            if rec.t < tnear {
                tnear = rec.t;
                closest = Some(rec);
            }
        }
    }
    if let Some(c) = closest {
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

//fn balls_demo() -> (Camera, Vec<Hittable>) {
//    let lookfrom = Vec3::new(3.0, 3.0, 2.0);
//    let lookat = Vec3::new(0.0, 0.0, -1.0);
//    let vup = Vec3::new(0.0, 1.0, 0.0);
//    let dist_to_focus = (lookfrom - lookat).length();
//    let aperture = 2.0;
//    let cam = Camera::new(
//        lookfrom, lookat, vup,
//        20.0, ASPECT_RATIO,
//        aperture, dist_to_focus);
//
//    (cam, 
//    vec![
//        Sphere::new(
//            Vec3::new(0.0,0.0,-1.0),
//            0.5,
//            Material::Lambertian(Lambertian::new(Vec3::new(0.1, 0.2, 0.5))),
//        ),
//        Sphere::new(
//            Vec3::new(0.0,-100.5,-1.0),
//            100.0,
//            Material::Lambertian(Lambertian::new(Vec3::new(0.8, 0.8, 0.0))),
//        ),
//        Sphere::new(
//            Vec3::new(1.0,0.0,-1.0),
//            0.5,
//            Material::Metal(Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.3)),
//        ),
//        Sphere::new(
//            Vec3::new(-1.0,0.0,-1.0),
//            0.5,
//            Material::Dielectric(Dielectric::new(1.5)),
//        ),
//        Sphere::new(
//            Vec3::new(-1.0,0.0,-1.0),
//            -0.45,
//            Material::Dielectric(Dielectric::new(1.5)),
//        ),
//    ])
//}

fn random_spheres_demo() -> Vec<Box<dyn Hittable+Send+Sync>> {
    let mut world : Vec<Box<dyn Hittable+Send+Sync>> = vec![];
    
    // Ground
    let ground_material = Material::Lambertian(
        Lambertian::new(Vec3::new_const(0.5))
    );
    world.push(Box::new(Sphere::new(Vec3::new(0.0,-1000.0,0.0), 1000.0, ground_material)));

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
                    let center2 = center + Vec3::new(0.0, rng.gen_range(0.0, 0.5), 0.0);
                    world.push(Box::new(
                        MovingSphere::new(
                            center, center2, 0.0, 1.0, 0.2,
                            Material::Lambertian(Lambertian::new(albedo))
                        ))
                    );
                }
                else if choose_mat < 0.95 {
                    let albedo = Vec3::random_range(0.5, 1.0);
                    let fuzz = rng.gen_range(0.0, 0.5);
                     world.push(Box::new(
                        Sphere::new(
                            center, 0.2,
                            Material::Metal(Metal::new(albedo,fuzz))
                        ))
                    );
                }
                else {
                    world.push(Box::new(
                        Sphere::new(
                            center, 0.2,
                            Material::Dielectric(Dielectric::new(1.5))
                        ))
                    );
                }
            }
        }
    }

    // Three big spheres
    world.push(Box::new(
        Sphere::new(
            Vec3::new(0.0, 1.0, 0.0), 1.0,
            Material::Dielectric(Dielectric::new(1.5))
        ))
    );
    world.push(Box::new(
        Sphere::new(
            Vec3::new(-4.0, 1.0, 0.0), 1.0,
            Material::Lambertian(Lambertian::new(Vec3::new(0.4, 0.2, 0.1)))
        ))
    );
    world.push(Box::new(
        Sphere::new(
            Vec3::new(4.0, 1.0, 0.0), 1.0,
            Material::Metal(Metal::new(Vec3::new(0.7, 0.6, 0.5), 0.0))
        ))
    );

    world
}

fn main() -> Result<(), std::io::Error> {
    let mut pixels: Vec<Vec3> = vec![Vec3::new_const(0.0); WIDTH * HEIGHT];

    // Camera and world
    // let (cam, world) = balls_demo();
    eprintln!("Generating scene...");
    let world = Arc::new(random_spheres_demo());

    let radius = 14.0;

    let mut angle = 15.0_f32;
    let mut file_idx = 0;
    while angle < 360.0 {
        // Set up camera
        let look_x = radius*angle.to_radians().cos();
        let look_z = radius*angle.to_radians().sin();
        let lookfrom = Vec3::new(look_x,2.0,look_z);
        let lookat = Vec3::new(0.0,0.0,0.0);
        let vup = Vec3::new(0.0,1.0,0.0);
        let dist_to_focus = 10.0;
        let aperture = 0.1;

        let cam = Camera::new(lookfrom, lookat, vup, 20.0, ASPECT_RATIO, aperture, dist_to_focus, 0.0, 1.0);

        // Trace rays
        pixels.par_iter_mut().enumerate().for_each(|(i,pix)| {
            let world_clone = world.clone();
            let x = i % WIDTH;
            let y = i / WIDTH;
            let mut rng = rand::thread_rng();
            let mut c = Vec3::new_const(0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = ((x as f32)+rng.gen::<f32>()) / ((WIDTH-1) as f32);
                let v = ((y as f32)+rng.gen::<f32>()) / ((HEIGHT-1) as f32);
                c += ray_color(cam.get_ray(u,v), &world_clone, 1);
            }
            c /= SAMPLES_PER_PIXEL as f32;
            *pix = c;
        });
        
        // Write output
        let filename = format!("output_{:04}.ppm", file_idx);
        eprintln!("Wrote frame {}...", filename);
        let file = File::create(filename).unwrap();
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

        angle += 0.5;
        file_idx += 1;
        break;
    }

    Ok(())
}
