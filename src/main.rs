#![allow(dead_code)]

#[macro_use]
extern crate derive_new;
extern crate rand;
extern crate rayon;

use accel::BVHNode;
use hittable::{Hittable, HittableSS};
use rand::Rng;
use rayon::prelude::*;
use scene::bowser_demo;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;
use util::random_in_unit_disk;
use vec3::Vec3;

mod accel;
mod hittable;
mod material;
mod scene;
mod util;
mod vec3;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: usize = 1024;
const HEIGHT: usize = ((WIDTH as f32) / ASPECT_RATIO) as usize;
const SAMPLES_PER_PIXEL: usize = 100;
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

fn ray_color(r: Ray, world: Arc<dyn Hittable>, depth: u32) -> Vec3 {
    let background = Vec3::new_const(0.0);

    if depth > MAX_DEPTH {
        return Vec3::new_const(0.0);
    }

    if let Some(c) = world.hit(&r, 0.001, f32::INFINITY) {
        let (did_scatter, attenuation, scattered) = c.material.scatter(r, &c);
        let emitted = c.material.emitted(c.u, c.v, c.p);
        if did_scatter {
            emitted + attenuation * ray_color(scattered, world, depth + 1)
        } else {
            emitted
        }
    } else {
        background
    }
}

fn main() -> Result<(), std::io::Error> {
    let mut pixels: Vec<Vec3> = vec![Vec3::new_const(0.0); WIDTH * HEIGHT];

    // Camera and world
    // let (cam, world) = balls_demo();
    eprintln!("Generating scene...");
    //random_spheres_demo(&mut world_vec);
    //balls_demo(&mut world_vec);
    //two_spheres_demo(&mut world_vec);
    //cornell_box(&mut world_vec);
    //final_scene(&mut world_vec);

    let radius = 20.0;

    let mut angle = 0.0_f32;
    let mut file_idx = 0;
    while file_idx < 1000 {
        let mut world_vec: Vec<Arc<HittableSS>> = vec![];
        bowser_demo(&mut world_vec, angle);
        let world_bvh = Arc::new(BVHNode::new(&mut world_vec[..]));
        // Timing
        let start = Instant::now();

        // Set up camera
        let look_x = radius * (35.0_f32).to_radians().cos();
        let look_z = radius * (35.0_f32).to_radians().sin();
        //let lookfrom = Vec3::new(look_x,1.5,look_z);
        let lookfrom = Vec3::new(look_x, 2.5, look_z);
        let lookat = Vec3::new(0.0, 2.0, 0.0);
        let vup = Vec3::new(0.0, 1.0, 0.0);
        let dist_to_focus = 10.0;
        let aperture = 0.0;
        let vfov = 20.0;

        /*
        let lookfrom = Vec3::new(478.0, 278.0, -600.0);
        let lookat = Vec3::new(278.0,278.0,0.0);
        let vup = Vec3::new(0.0,1.0,0.0);
        let dist_to_focus = 10.0;
        let aperture = 0.0;
        let vfov = 40.0;
        */

        let cam = Camera::new(
            lookfrom,
            lookat,
            vup,
            vfov,
            ASPECT_RATIO,
            aperture,
            dist_to_focus,
            0.0,
            1.0,
        );

        // Trace rays
        pixels.par_iter_mut().enumerate().for_each(|(i, pix)| {
            let x = i % WIDTH;
            let y = i / WIDTH;
            let mut rng = rand::thread_rng();
            let mut c = Vec3::new_const(0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = ((x as f32) + rng.gen::<f32>()) / ((WIDTH - 1) as f32);
                let v = ((y as f32) + rng.gen::<f32>()) / ((HEIGHT - 1) as f32);
                let ray = cam.get_ray(u, v);
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

        for y in { 0..HEIGHT }.rev() {
            for x in 0..WIDTH {
                let (ir, ig, ib) = pixels[y * WIDTH + x].to_color();
                writeln!(&mut wr, "{} {} {}", ir, ig, ib)?;
            }
        }
        eprintln!("Wrote frame {} in {}s", filename, start.elapsed().as_secs());

        angle += 0.5;
        file_idx += 1;
    }

    Ok(())
}
