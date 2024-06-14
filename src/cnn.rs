use ndarray::{Array, Array3, Array2, s, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;

pub struct CNN {
    w_con: Array3<f32>,
    w_fc: Array2<f32>,
}

impl CNN {
    pub fn new() -> Self {
        let w_con = Array::random((3, 3, 8), Standard) / 9.0;
        let w_fc = Array::random((1352, 10), Standard);
        CNN { w_con, w_fc }
    }

    pub fn train(&mut self, images: &Array3<f32>, labels: &Vec<u8>, learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for (i, image) in images.outer_iter().enumerate() {
                let label = labels[i];
                let image = image.to_owned().into_shape((1, 28, 28)).unwrap(); // Ensure the image is reshaped correctly
                let output = self.forward(&image);
                let mut target = Array2::<f32>::zeros((1, 10));
                target[[0, label as usize]] = 1.0;
                epoch_loss += self.cross_entropy_loss(&output, &target);
                self.backward(&image, &output, &target, learning_rate);
            }
            println!("Epoch {}: Loss = {:.6}", epoch + 1, epoch_loss / images.len_of(Axis(0)) as f32);
        }
    }

    pub fn test(&self, images: &Array3<f32>, labels: &Vec<u8>) -> f32 {
        let mut correct = 0;
        for (i, image) in images.outer_iter().enumerate() {
            let label = labels[i];
            let image = image.to_owned().into_shape((1, 28, 28)).unwrap(); // Ensure the image is reshaped correctly
            let output = self.forward(&image);
            let predicted = output.index_axis(Axis(0), 0).iter().cloned().enumerate().max_by(|&a, &b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
            if predicted == label as usize {
                correct += 1;
            }
        }
        correct as f32 / images.len_of(Axis(0)) as f32
    }

    fn forward(&self, input: &Array3<f32>) -> Array2<f32> {
        let conv_output = self.convolution(input);
        let pooled_output = self.max_pooling(&conv_output);
        let fc_output = self.fully_connected(&pooled_output);
        self.softmax(&fc_output)
    }

    fn convolution(&self, input: &Array3<f32>) -> Array3<f32> {
        let (input_h, input_w, _) = input.dim(); // Remove unused input_d binding
        let filter_size = 3;
        let num_filters = 8;
        let output_h = input_h - filter_size + 1;
        let output_w = input_w - filter_size + 1;

        let mut output = Array3::<f32>::zeros((output_h, output_w, num_filters));

        for filter in 0..num_filters {
            for i in 0..output_h {
                for j in 0..output_w {
                    let region = input.slice(s![i..i + filter_size, j..j + filter_size, ..]);
                    output[[i, j, filter]] = (region * &self.w_con.slice(s![.., .., filter])).sum();
                }
            }
        }
        output
    }

    fn max_pooling(&self, input: &Array3<f32>) -> Array3<f32> {
        let (input_h, input_w, input_d) = input.dim();
        let pool_size = 2;
        let output_h = input_h / pool_size;
        let output_w = input_w / pool_size;

        let mut output = Array3::<f32>::zeros((output_h, output_w, input_d));

        for d in 0..input_d {
            for i in 0..output_h {
                for j in 0..output_w {
                    let region = input.slice(s![i * pool_size..(i + 1) * pool_size, j * pool_size..(j + 1) * pool_size, d]);
                    output[[i, j, d]] = region.iter().cloned().fold(f32::MIN, f32::max);
                }
            }
        }
        output
    }

    fn fully_connected(&self, input: &Array3<f32>) -> Array2<f32> {
        let flat_input = input.iter().cloned().collect::<Vec<f32>>();
        let input_array = Array::from(flat_input).into_shape((1, 1352)).unwrap();
        input_array.dot(&self.w_fc)
    }

    fn softmax(&self, input: &Array2<f32>) -> Array2<f32> {
        let exp_input = input.mapv(f32::exp);
        let sum_exp = exp_input.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp_input / sum_exp
    }

    fn cross_entropy_loss(&self, predicted: &Array2<f32>, actual: &Array2<f32>) -> f32 {
        -(&actual * &predicted.mapv(f32::ln)).sum()
    }

    fn backward(&mut self, input: &Array3<f32>, predicted: &Array2<f32>, actual: &Array2<f32>, learning_rate: f32) {
        // Compute loss gradient with respect to the output
        let mut dl_dy = predicted - actual; // Change to snake_case

        // Fully connected layer gradients
        let flat_input = input.iter().cloned().collect::<Vec<f32>>();
        let input_array = Array::from(flat_input).into_shape((1, 1352)).unwrap();
        let dl_dw_fc = input_array.t().dot(&dl_dy); // Change to snake_case
        let dl_dinput_fc = dl_dy.dot(&self.w_fc.t()); // Change to snake_case

        // Update fully connected layer weights
        self.w_fc -= &dl_dw_fc * learning_rate;

        // Max pooling layer gradients
        let conv_output = self.convolution(input);
        let pooled_output = self.max_pooling(&conv_output);
        let pool_h = pooled_output.shape()[0];
        let pool_w = pooled_output.shape()[1];
        let pool_d = pooled_output.shape()[2];

        let mut dl_dpool = Array3::<f32>::zeros(pooled_output.dim()); // Change to snake_case

        for d in 0..pool_d {
            for i in 0..pool_h {
                for j in 0..pool_w {
                    let pool_region = pooled_output.slice(s![i, j, d]);
                    let max_val = pool_region.iter().cloned().fold(f32::MIN, f32::max);
                    let max_mask = pool_region.mapv(|x| if x == max_val { 1.0 } else { 0.0 });
                    dl_dpool.slice_mut(s![i, j, d]).assign(&(&max_mask * dl_dinput_fc[[0, i * pool_w + j]])); // Change to snake_case
                }
            }
        }

        // Convolution layer gradients
        let (input_h, input_w, _) = input.dim(); // Remove unused input_d binding
        let filter_size = 3;
        let num_filters = 8;
        let output_h = input_h - filter_size + 1;
        let output_w = input_w - filter_size + 1;

        let mut dl_dw_con = Array3::<f32>::zeros(self.w_con.dim()); // Change to snake_case

        for filter in 0..num_filters {
            for i in 0..output_h {
                for j in 0..output_w {
                    let region = input.slice(s![i..i + filter_size, j..j + filter_size, ..]);
                    dl_dw_con.slice_mut(s![.., .., filter]).assign(&(&region * dl_dpool[[i, j, filter]])); // Change to snake_case
                }
            }
        }

        // Update convolutional layer weights
        self.w_con -= &dl_dw_con * learning_rate;
    }
}
