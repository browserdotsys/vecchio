use crate::rand::Rng;

#[derive(Debug, Copy, Clone, new)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new_const(x: f32) -> Vec3 {
        Vec3 { x, y: x, z: x }
    }

    pub fn zero() -> Vec3 {
        Vec3::new_const(0.0)
    }

    pub fn dot(&self, v: Vec3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn cross(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }

    pub fn length2(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f32 {
        self.length2().sqrt()
    }

    pub fn unit_vector(&self) -> Vec3 {
        let norm = self.length2().sqrt();
        Vec3::new(
            self.x / norm,
            self.y / norm,
            self.z / norm,
        )
    }

    pub fn clamp(x: f32, min: f32, max: f32) -> f32 {
        if x < min {
            min
        }
        else if x > max {
            max
        }
        else {
            x
        }
    }

    pub fn to_color(&self) -> (u32,u32,u32) {
        // Note: sqrt() here does gamma correction
        (
            (256.0 * Vec3::clamp(self.x.sqrt(), 0.0, 0.999)) as u32,
            (256.0 * Vec3::clamp(self.y.sqrt(), 0.0, 0.999)) as u32,
            (256.0 * Vec3::clamp(self.z.sqrt(), 0.0, 0.999)) as u32,
        )
    }

    #[allow(dead_code)]
    pub fn powf(&self, exp: f32) -> Vec3 {
        Vec3::new(self.x.powf(exp), self.y.powf(exp), self.z.powf(exp))
    }

    pub fn random() -> Vec3 {
        let mut rng = rand::thread_rng();

        Vec3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>())
    }

    pub fn random_range(min: f32, max: f32) -> Vec3 {
        let mut rng = rand::thread_rng();

        Vec3::new(
            rng.gen_range(min, max),
            rng.gen_range(min, max),
            rng.gen_range(min, max),
        )
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, t: Vec3) {
        *self = Self {
            x: self.x + t.x,
            y: self.y + t.y,
            z: self.z + t.z,
        };
    }
}

impl std::ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, t: f32) {
        *self = Self {
            x: self.x * t,
            y: self.y * t,
            z: self.z * t,
        };
    }
}

impl std::ops::DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, t: f32) {
        *self = Self {
            x: self.x / t,
            y: self.y / t,
            z: self.z / t,
        };
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => unreachable!(),
        }
    }
}

impl std::ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => unreachable!(),
        }
    }
}

impl std::cmp::PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}
