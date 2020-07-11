use rand::Rng;
use std::sync::Arc;
use crate::hittable::{Sphere,MovingSphere,Rect,Boxy,Translate,
               RotateX,RotateY,RotateZ,FlipFace,
               ConstantMedium,Hittable,HittableSS};
use crate::material::{Lambertian,SolidColor,Metal,Dielectric,
               Checker,ImageTexture,NoiseTexture,
               DiffuseLight};
use crate::accel::{BVHNode,AxisBB};
use crate::vec3::Vec3;
use crate::hittable::HitRec;
use crate::Ray;

pub fn balls_demo(world: &mut Vec<Arc<HittableSS>>) {
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

pub fn random_spheres_demo(world: &mut Vec<Arc<HittableSS>>) {
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

pub fn two_spheres_demo(world: &mut Vec<Arc<HittableSS>>) {
    // Ground
    let pertext = Arc::new(Lambertian::new(Arc::new(NoiseTexture::new(2.0))));
    world.push(Arc::new(Sphere::new(Vec3::new(0.0,-1000.0,0.0), 1000.0, pertext.clone())));

    world.push(Arc::new(
        Sphere::new(
            Vec3::new(0.0, 2.0, 0.0), 2.0,
            pertext,
        )
    ));

    // Light
    world.push(Arc::new(
        Rect::XZRect(-6.0, 6.0, -6.0, 6.0, 8.0,
            Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Vec3::new_const(4.0)))))
        )
    ));
}

struct Bowser {
    parts: Arc<HittableSS>,
}

impl Bowser {
    pub fn new(x: f32, y: f32, z: f32) -> Bowser {
        let mut world : Vec<Arc<HittableSS>> = vec![];
        // Face
        world.push(Arc::new(
            Rect::XYRect(x-2.0, x+2.0, y-1.875+1.0, y-1.875+4.0, z+4.5-3.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_face.png"))))
            )
        ));
        // Top
        world.push(Arc::new(
            Rect::XZRect(x-2.0, x+2.0, z+4.5-6.0, z+4.5-3.0, y-1.875+4.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_top.png"))))
            )
        ));
        // Back
        world.push(Arc::new(
            Rect::XYRect(x-2.0, x+2.0, y-1.875+1.0, y-1.875+4.0, z+4.5-6.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_back.png"))))
            )
        ));
        // Sides
        world.push(Arc::new(
            Rect::YZRect(y-1.875+1.0, y-1.875+4.0, z+4.5-6.0, z+4.5-3.0, x-2.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_side.png"))))
            )
        ));
        world.push(Arc::new(
            Rect::YZRect(y-1.875+1.0, y-1.875+4.0, z+4.5-6.0, z+4.5-3.0, x+2.0,
                Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/bowser_side.png"))))
            )
        ));
        // Bottom
        let grey = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.278, 0.387, 0.438)))));
        world.push(Arc::new(
            Rect::XZRect(x-2.0, x+2.0, z+4.5-6.0, z+4.5-3.0, y-1.875+1.0, grey.clone()
            )
        ));
        // Feet
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.5, y-1.875+0.5, z+4.5-4.75),
                Vec3::new(x-0.5, y-1.875+1.0, z+4.5-4.25),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.5, y-1.875+0.5, z+4.5-4.75),
                Vec3::new(x+1.5, y-1.875+1.0, z+4.5-4.25),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.5, y-1.875+0.25, z+4.5-4.75),
                Vec3::new(x-0.5, y-1.875+0.5, z+4.5-3.5),
                grey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.5, y-1.875+0.25, z+4.5-4.75),
                Vec3::new(x+1.5, y-1.875+0.5, z+4.5-3.5),
                grey,
            )
        ));
        // Arms
        let brown = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.4, 0.2, 0.1)))));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.25, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x-2.00, y-1.875+2.75, z+4.5-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.50, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x-2.25, y-1.875+2.50, z+4.5-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.75, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x-2.50, y-1.875+2.25, z+4.5-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.00, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x+2.25, y-1.875+2.75, z+4.5-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.25, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x+2.50, y-1.875+2.50, z+4.5-4.35),
                brown.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+2.50, y-1.875+1.75, z+4.5-4.65),
                Vec3::new(x+2.75, y-1.875+2.25, z+4.5-4.35),
                brown,
            )
        ));
        // Face rim
        let lightgrey = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.601, 0.687, 0.723)))));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y-1.875+3.875, z+4.5-3.00),
                Vec3::new(x+2.0, y-1.875+4.00, z+4.5-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y-1.875+1.0, z+4.5-3.00),
                Vec3::new(x+2.0, y-1.875+1.125, z+4.5-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-2.0, y-1.875+1.125, z+4.5-3.00),
                Vec3::new(x-1.875, y-1.875+3.875, z+4.5-2.875),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+1.875, y-1.875+1.125, z+4.5-3.00),
                Vec3::new(x+2.0, y-1.875+3.875, z+4.5-2.875),
                lightgrey.clone(),
            )
        ));

        // Butt ports
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y-1.875+1.625, z+4.5-6.125),
                Vec3::new(x-0.875, y-1.875+1.75, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y-1.875+1.125, z+4.5-6.125),
                Vec3::new(x-0.875, y-1.875+1.25, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.875, y-1.875+1.25, z+4.5-6.125),
                Vec3::new(x-1.750, y-1.875+1.625, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x-1.0, y-1.875+1.25, z+4.5-6.125),
                Vec3::new(x-0.875, y-1.875+1.625, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));

        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y-1.875+1.625, z+4.5-6.125),
                Vec3::new(x+1.875, y-1.875+1.75, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y-1.875+1.125, z+4.5-6.125),
                Vec3::new(x+1.875, y-1.875+1.25, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+1.750, y-1.875+1.25, z+4.5-6.125),
                Vec3::new(x+1.875, y-1.875+1.625, z+4.5-6.0),
                lightgrey.clone(),
            )
        ));
        world.push(Arc::new(
            Boxy::new(
                Vec3::new(x+0.875, y-1.875+1.25, z+4.5-6.125),
                Vec3::new(x+1.0, y-1.875+1.625, z+4.5-6.0),
                lightgrey,
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

pub fn bowser_demo(world: &mut Vec<Arc<HittableSS>>, angle: f32) {
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
    // off the ground, so fix that with a Translate here. Rotate him too.
    world.push(
        Arc::new(Translate::new(
            Arc::new(RotateX::new(
            Arc::new(RotateY::new(
            Arc::new(RotateZ::new(
                Arc::new(Bowser::new(0.0,0.0,0.0)),
                angle)), angle)), angle)),
            Vec3::new(0.0, 1.625, -4.5),
        ))
    );

    // Light
    world.push(Arc::new(
        Rect::XYRect(-2.0, 2.0, 1.0, 4.0, 3.0,
            Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Vec3::new_const(4.0)))))
        )
    ));

    /* [mirrors removed for now]
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
    */
}

pub fn cornell_box(world: &mut Vec<Arc<HittableSS>>) {
    let red   = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.65, 0.05, 0.05)))));
    let white = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.73, 0.73, 0.73)))));
    let green = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.12, 0.45, 0.15)))));
    let light = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Vec3::new(15.0, 15.0, 15.0)))));
    world.push(Arc::new(FlipFace::new(Arc::new(Rect::YZRect(0.0, 555.0, 0.0, 555.0, 555.0, green)))));
    world.push(Arc::new(Rect::YZRect(0.0, 555.0, 0.0, 555.0, 0.0, red)));
    world.push(Arc::new(Rect::XZRect(213.0, 343.0, 227.0, 332.0, 554.0, light)));
    world.push(Arc::new(FlipFace::new(Arc::new(Rect::XZRect(0.0, 555.0, 0.0, 555.0, 0.0, white.clone())))));
    world.push(Arc::new(Rect::XZRect(0.0, 555.0, 0.0, 555.0, 555.0, white.clone())));
    world.push(Arc::new(FlipFace::new(Arc::new(Rect::XYRect(0.0, 555.0, 0.0, 555.0, 555.0, white.clone())))));

    let box1 = Arc::new(Boxy::new(Vec3::new_const(0.0), Vec3::new(165.0, 330.0, 165.0), white.clone()));
    world.push(
        Arc::new(Translate::new(
                Arc::new(RotateY::new(box1, 15.0)),
                Vec3::new(265.0, 0.0, 295.0),
        )),
    );
    let box2 = Arc::new(Boxy::new(Vec3::new_const(0.0), Vec3::new(165.0, 165.0, 165.0), white));
    world.push(
        Arc::new(Translate::new(
                Arc::new(RotateY::new(box2, -18.0)),
                Vec3::new(130.0, 0.0, 65.0),
        )),
    );
}

pub fn final_scene(objects: &mut Vec<Arc<HittableSS>>) {
    let mut rng = rand::thread_rng();
    let mut boxes1 : Vec<Arc<HittableSS>> = vec![];
    let ground = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.48, 0.83, 0.53)))));

    const BOXES_PER_SIDE : usize = 20;

    for i in 0..BOXES_PER_SIDE {
        for j in 0..BOXES_PER_SIDE {
            let w = 100.0;
            let x0 = -1000.0 + (i as f32)*w;
            let z0 = -1000.0 + (j as f32)*w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let z1 = z0 + w;
            let y1 = rng.gen_range(1.0, 101.0);
            boxes1.push(Arc::new(Boxy::new(
                        Vec3::new(x0, y0, z0),
                        Vec3::new(x1, y1, z1),
                        ground.clone()
            )));
        }
    }

    objects.push(Arc::new(BVHNode::new(&mut boxes1)));

    let light = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Vec3::new_const(7.0)))));
    objects.push(Arc::new(Rect::XZRect(123.0, 423.0, 147.0, 412.0, 554.0, light)));

    let center1 = Vec3::new(400.0, 400.0, 200.0);
    let center2 = center1 + Vec3::new(30.0,0.0,0.0);
    let moving_sphere_material = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new(0.7, 0.3, 0.1)))));
    objects.push(Arc::new(MovingSphere::new(center1, center2, 0.0, 1.0, 50.0, moving_sphere_material)));

    objects.push(Arc::new(Sphere::new(Vec3::new(260.0, 150.0, 45.0), 50.0, Arc::new(Dielectric::new(1.5)))));
    objects.push(Arc::new(Sphere::new(
        Vec3::new(0.0, 150.0, 145.0), 50.0,
        Arc::new(Metal::new(Arc::new(SolidColor::new(Vec3::new(0.8, 0.8, 0.9))), 10.0))
    )));

    let boundary1 = Arc::new(Sphere::new(Vec3::new(360.0,150.0,145.0), 70.0, Arc::new(Dielectric::new(1.5))));
    objects.push(boundary1.clone());
    objects.push(Arc::new(ConstantMedium::new(
        boundary1, 0.2, Arc::new(SolidColor::new(Vec3::new(0.2, 0.4, 0.9)))
    )));
    let boundary2 = Arc::new(Sphere::new(Vec3::new_const(0.0), 5000.0, Arc::new(Dielectric::new(1.5))));
    objects.push(Arc::new(ConstantMedium::new(
        boundary2, 0.0001, Arc::new(SolidColor::new(Vec3::new_const(1.0))))
    ));

    let emat = Arc::new(Lambertian::new(Arc::new(ImageTexture::new("assets/earthmap.png"))));
    objects.push(Arc::new(Sphere::new(Vec3::new(400.0,200.0,400.0), 100.0, emat)));
    let pertext = Arc::new(NoiseTexture::new(0.1));
    objects.push(Arc::new(Sphere::new(Vec3::new(220.0,280.0,300.0), 80.0, Arc::new(Lambertian::new(pertext)))));
    
    let mut boxes2 : Vec<Arc<HittableSS>> = vec![];
    let white = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Vec3::new_const(0.73)))));
    for _ in 0..1000 {
        boxes2.push(Arc::new(Sphere::new(Vec3::random_range(0.0,165.0), 10.0, white.clone())));
    }

    objects.push(Arc::new(
        Translate::new(
            Arc::new(RotateY::new(Arc::new(BVHNode::new(&mut boxes2)), 15.0)),
            Vec3::new(-100.0,270.0,395.0),
        )
    ));
}
