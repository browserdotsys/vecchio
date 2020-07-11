use crate::vec3::Vec3;
use std::sync::Arc;
use crate::util::{fmin,fmax};
use crate::hittable::{Hittable,HittableSS,HitRec};
use crate::Ray;
use crate::rand::Rng;

// Axis-aligned bounding box
#[derive(new,Copy,Clone)]
pub struct AxisBB {
    pub min: Vec3,
    pub max: Vec3,
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
        true
    }

    pub fn surrounding_box(box1: AxisBB, box2: AxisBB) -> AxisBB {
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

pub struct BVHNode {
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
                    rec_left
                }
                else {
                    rec_right
                }
            },
            (Some(_), None) => {
                rec_left
            },
            (None, Some(_)) => {
                rec_right
            },
            (None, None) => {
                None
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

    pub fn new(objects: &mut [Arc<HittableSS>]) -> BVHNode {
        let mut rng = rand::thread_rng();
        let axis = rng.gen_range(0,3);

        if objects.len() == 1 {
            BVHNode {
                left: objects[0].clone(),
                right: objects[0].clone(),
                bb: BVHNode::box_for_hittables(&objects[0], &objects[0]),
            }
        }
        else if objects.len() == 2 {
            let a_bb = objects[0].bounding_box(0.0, 0.0).unwrap();
            let b_bb = objects[1].bounding_box(0.0, 0.0).unwrap();
            let (mut i1, mut i2) = (0,1);
            if a_bb.min[axis] < b_bb.min[axis] {
                i1 = 1;
                i2 = 0;
            }
            BVHNode {
                left: objects[i1].clone(),
                right: objects[i2].clone(),
                bb: BVHNode::box_for_hittables(&objects[i1], &objects[i2]),
            }
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
            BVHNode { left, right, bb }
        }
    }
}
