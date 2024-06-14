mod idx_reader;
mod cnn;

use cnn::CNN;
use idx_reader::{process_idx_files, read_idx_images, read_idx_labels};
use std::fs;
use ndarray::{Array3, s};
use std::io;

fn main() -> io::Result<()> {
    let image_file_path = "images.idx3-ubyte";
    let label_file_path = "labels.idx1-ubyte";
    let output_dir = "output_images";

    // Check if the output images have been generated
    if !fs::metadata(output_dir).is_ok() {
        // Generate images if not generated
        process_idx_files(image_file_path, label_file_path, output_dir)?;
        println!("Images have been generated and saved to {}", output_dir);
        return Ok(());
    } else {
        println!("Images already generated, proceeding to CNN training and testing...");
    }

    // Load data for CNN training and testing
    let (train_images, train_labels, test_images, test_labels) = load_data(image_file_path, label_file_path)?;

    let mut cnn = CNN::new();
    cnn.train(&train_images, &train_labels, 0.005, 10);  // Reduced epochs for quick validation
    let accuracy = cnn.test(&test_images, &test_labels);

    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

// Function to load data from IDX files using idx_reader module
fn load_data(image_path: &str, label_path: &str) -> io::Result<(Array3<f32>, Vec<u8>, Array3<f32>, Vec<u8>)> {
    let images = read_idx_images(image_path)?;
    let labels = read_idx_labels(label_path)?;

    // Convert images to Array3<f32> and normalize
    let num_images = images.len();
    let num_rows = 28;
    let num_cols = 28;
    let mut image_array = Array3::<f32>::zeros((num_images, num_rows, num_cols));

    for (i, image) in images.iter().enumerate() {
        for (j, &pixel) in image.iter().enumerate() {
            let row = j / num_cols;
            let col = j % num_cols;
            image_array[[i, row, col]] = pixel as f32 / 255.0;
        }
    }

    // Split into training and testing datasets
    let train_size = 1000;
    let train_images = image_array.slice(s![0..train_size, .., ..]).to_owned();
    let train_labels = labels[0..train_size].to_vec();
    let test_images = image_array.slice(s![train_size.., .., ..]).to_owned();
    let test_labels = labels[train_size..].to_vec();

    Ok((train_images, train_labels, test_images, test_labels))
}