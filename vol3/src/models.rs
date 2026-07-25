use crate::layers::{Conv2d, Layer, Linear, Parameter};
use crate::variable::Variable;

pub struct VGG16 {
    pub conv1_1: Conv2d,
    pub conv1_2: Conv2d,
    pub conv2_1: Conv2d,
    pub conv2_2: Conv2d,
    pub conv3_1: Conv2d,
    pub conv3_2: Conv2d,
    pub conv3_3: Conv2d,
    pub conv4_1: Conv2d,
    pub conv4_2: Conv2d,
    pub conv4_3: Conv2d,
    pub conv5_1: Conv2d,
    pub conv5_2: Conv2d,
    pub conv5_3: Conv2d,
    pub fc6: Linear,
    pub fc7: Linear,
    pub fc8: Linear,
}

impl VGG16 {
    pub fn new(rng: &mut impl rand::Rng) -> Self {
        Self {
            conv1_1: Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), false, rng),
            conv1_2: Conv2d::new(64, 64, (3, 3), (1, 1), (1, 1), false, rng),
            conv2_1: Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), false, rng),
            conv2_2: Conv2d::new(128, 128, (3, 3), (1, 1), (1, 1), false, rng),
            conv3_1: Conv2d::new(128, 256, (3, 3), (1, 1), (1, 1), false, rng),
            conv3_2: Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), false, rng),
            conv3_3: Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), false, rng),
            conv4_1: Conv2d::new(256, 512, (3, 3), (1, 1), (1, 1), false, rng),
            conv4_2: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), false, rng),
            conv4_3: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), false, rng),
            conv5_1: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), false, rng),
            conv5_2: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), false, rng),
            conv5_3: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), false, rng),
            fc6: Linear::new(512 * 7 * 7, 4096, false, rng),
            fc7: Linear::new(4096, 4096, false, rng),
            fc8: Linear::new(4096, 1000, false, rng),
        }
    }
}

impl Layer for VGG16 {
    fn params(&self) -> Vec<Parameter> {
        self.named_params().into_iter().map(|(_, p)| p).collect()
    }

    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut p = Vec::new();
        let layers: Vec<(&str, &dyn Layer)> = vec![
            ("conv1_1", &self.conv1_1),
            ("conv1_2", &self.conv1_2),
            ("conv2_1", &self.conv2_1),
            ("conv2_2", &self.conv2_2),
            ("conv3_1", &self.conv3_1),
            ("conv3_2", &self.conv3_2),
            ("conv3_3", &self.conv3_3),
            ("conv4_1", &self.conv4_1),
            ("conv4_2", &self.conv4_2),
            ("conv4_3", &self.conv4_3),
            ("conv5_1", &self.conv5_1),
            ("conv5_2", &self.conv5_2),
            ("conv5_3", &self.conv5_3),
            ("fc6", &self.fc6),
            ("fc7", &self.fc7),
            ("fc8", &self.fc8),
        ];

        for (name, l) in layers {
            for (sub_name, param) in l.named_params() {
                p.push((format!("{}/{}", name, sub_name), param));
            }
        }
        p
    }

    fn forward(&self, x: &Variable) -> Variable {
        let mut h = x.clone();

        h = crate::functions::relu(&self.conv1_1.forward(&h));
        h = crate::functions::relu(&self.conv1_2.forward(&h));
        h = h.pooling_simple((2, 2), (2, 2), (0, 0));

        h = crate::functions::relu(&self.conv2_1.forward(&h));
        h = crate::functions::relu(&self.conv2_2.forward(&h));
        h = h.pooling_simple((2, 2), (2, 2), (0, 0));

        h = crate::functions::relu(&self.conv3_1.forward(&h));
        h = crate::functions::relu(&self.conv3_2.forward(&h));
        h = crate::functions::relu(&self.conv3_3.forward(&h));
        h = h.pooling_simple((2, 2), (2, 2), (0, 0));

        h = crate::functions::relu(&self.conv4_1.forward(&h));
        h = crate::functions::relu(&self.conv4_2.forward(&h));
        h = crate::functions::relu(&self.conv4_3.forward(&h));
        h = h.pooling_simple((2, 2), (2, 2), (0, 0));

        h = crate::functions::relu(&self.conv5_1.forward(&h));
        h = crate::functions::relu(&self.conv5_2.forward(&h));
        h = crate::functions::relu(&self.conv5_3.forward(&h));
        h = h.pooling_simple((2, 2), (2, 2), (0, 0));

        h = h.reshape(&[h.shape()[0], 25088]);

        h = crate::functions::relu(&self.fc6.forward(&h)).dropout(0.5);
        h = crate::functions::relu(&self.fc7.forward(&h)).dropout(0.5);
        h = self.fc8.forward(&h);

        h
    }
}
