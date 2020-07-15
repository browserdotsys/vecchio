#![allow(dead_code)]

#[macro_use]
extern crate derive_new;
extern crate rand;
extern crate rayon;

use accel::BVHNode;
use hittable::HittableSS;
use rand::Rng;
use rayon::prelude::*;
use scene::{bowser_demo,cornell_box,final_scene};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;
use util::{random_in_unit_disk,HittablePDF,MixturePDF,PDF};
use vec3::Vec3;

mod accel;
mod hittable;
mod material;
mod scene;
mod util;
mod vec3;

const SAMPLES_PER_PIXEL: usize = 1000;
const MAX_DEPTH: u32 = 100;

#[derive(Copy, Clone)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3,
    time: f32,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray::new_with_time(origin, direction, 0.0)
    }

    fn new_with_time(origin: Vec3, direction: Vec3, time: f32) -> Ray {
        Ray {
            origin,
            direction,
            time,
        }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Copy, Clone)]
pub struct Camera {
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
    fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        vup: Vec3,
        vfov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
        time0: f32,
        time1: f32,
    ) -> Camera {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = h * 2.0;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).unit_vector();
        let u = vup.cross(w).unit_vector();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;
        let lens_radius = aperture / 2.0;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius,
            time0,
            time1,
        }
    }

    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let mut rng = rand::thread_rng();
        let rd = random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new_with_time(
            self.origin + offset,
            self.lower_left_corner + self.horizontal * s + self.vertical * t - self.origin - offset,
            rng.gen_range(self.time0, self.time1),
        )
    }
}

fn ray_color(r: Ray, world: Arc<HittableSS>, lights: Arc<HittableSS>, depth: u32) -> Vec3 {
    let background = Vec3::new_const(0.0);

    if depth > MAX_DEPTH {
        return Vec3::new_const(0.0);
    }

    if let Some(c) = world.hit(&r, 0.001, f32::INFINITY) {
        let emitted = c.material.emitted(&c, c.u, c.v, c.p);
        if let Some(srec) = c.material.scatter_with_pdf(r, &c) {
            // Specular!
            if let Some(spec) = srec.specular_ray {
                return srec.attenuation *
                    ray_color(spec, world, lights, depth + 1);
            }

            let p_important = HittablePDF::new(lights.clone(), c.p);
            let p = MixturePDF::new(Arc::new(p_important), 0.5, srec.pdf.clone(), 0.5);

            let scattered = Ray::new_with_time(c.p, p.generate(), r.time);
            let pdf = p.value(scattered.direction);
            emitted +
                srec.attenuation * c.material.scattering_pdf(r, &c, scattered) *
                ray_color(scattered, world, lights, depth + 1) / pdf
        } else {
            emitted
        }
    } else {
        background
    }
}

fn main() -> Result<(), std::io::Error> {
    // Camera and world
    // let (cam, world) = balls_demo();
    eprintln!("Generating scene...");
    //random_spheres_demo(&mut world_vec);
    //balls_demo(&mut world_vec);
    //two_spheres_demo(&mut world_vec);
    //let mut world_vec: Vec<Arc<HittableSS>> = vec![];
    //let (mut world_vec, cam_iter) = cornell_box();
    //final_scene(&mut world_vec);

    let mut config = match 2 {
        0 => bowser_demo(),
        1 => cornell_box(),
        2 => final_scene(),
        _ => panic!("Not a valid scene"),
    };
    //let mut config = cornell_box();
    let world_bvh = Arc::new(BVHNode::new(&mut config.world[..]));
    let important = Arc::new(config.lights);

    let width: usize = 600;
    let height: usize = ((width as f32) / config.aspect_ratio) as usize;
    let mut pixels: Vec<Vec3> = vec![Vec3::new_const(0.0); width * height];

    let mut file_idx = 0;
    for cam in config.cam_iter {
        // Timing
        let start = Instant::now();

        // Trace rays
        pixels.par_iter_mut().enumerate().for_each(|(i, pix)| {
            let x = i % width;
            let y = i / width;
            let mut rng = rand::thread_rng();
            let mut c = Vec3::new_const(0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = ((x as f32) + rng.gen::<f32>()) / ((width - 1) as f32);
                let v = ((y as f32) + rng.gen::<f32>()) / ((height - 1) as f32);
                let ray = cam.get_ray(u, v);
                let color = ray_color(ray, world_bvh.clone(), important.clone(), 1);
                // Filter NaN and infinity
                if color.x.is_finite() && color.y.is_finite() && color.z.is_finite() {
                    c += color;
                }
            }
            c /= SAMPLES_PER_PIXEL as f32;
            *pix = c;
        });

        // Write output
        let filename = format!("output_{:04}.ppm", file_idx);
        let file = File::create(filename.clone()).unwrap();
        let mut wr = BufWriter::new(&file);

        writeln!(&mut wr, "P3")?;
        writeln!(&mut wr, "{} {}", width, height)?;
        writeln!(&mut wr, "255")?;

        for y in { 0..height }.rev() {
            for x in 0..width {
                let (ir, ig, ib) = pixels[y * width + x].to_color();
                writeln!(&mut wr, "{} {} {}", ir, ig, ib)?;
            }
        }
        eprintln!("Wrote frame {} in {}s", filename, start.elapsed().as_secs());

        file_idx += 1;
    }

    Ok(())
}
