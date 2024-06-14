use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;
use image::{GrayImage, Luma};

pub fn process_idx_files(image_file_path: &str, label_file_path: &str, output_dir: &str) -> io::Result<()> {
    // Check if the output directory exists
    if Path::new(output_dir).exists() {
        println!("Directory {} already exists. Skipping image generation.", output_dir);
        return Ok(());
    }

    // Create the output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    let images = read_idx_images(image_file_path)?;
    let labels = read_idx_labels(label_file_path)?;

    for (i, image) in images.iter().enumerate() {
        let label = labels[i];
        let file_name = format!("output_image_{}_label_{}.png", i, label);
        let file_path = Path::new(output_dir).join(file_name);
        save_image(&image, &file_path)?;
    }

    Ok(())
}

fn read_idx_images<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u8>>> {
    let mut file = File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    let magic_number = u32::from_be_bytes([contents[0], contents[1], contents[2], contents[3]]);
    let num_images = u32::from_be_bytes([contents[4], contents[5], contents[6], contents[7]]);
    let num_rows = u32::from_be_bytes([contents[8], contents[9], contents[10], contents[11]]);
    let num_cols = u32::from_be_bytes([contents[12], contents[13], contents[14], contents[15]]);

    assert_eq!(magic_number, 2051); // Magic number for IDX images

    let mut images = Vec::with_capacity(num_images as usize);
    let image_size = (num_rows * num_cols) as usize;
    for i in 0..num_images {
        let start = 16 + i as usize * image_size;
        let end = start + image_size;
        images.push(contents[start..end].to_vec());
    }

    Ok(images)
}

fn read_idx_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    let magic_number = u32::from_be_bytes([contents[0], contents[1], contents[2], contents[3]]);
    let num_labels = u32::from_be_bytes([contents[4], contents[5], contents[6], contents[7]]);

    assert_eq!(magic_number, 2049); // Magic number for IDX labels

    Ok(contents[8..(8 + num_labels as usize)].to_vec())
}

fn save_image<P: AsRef<Path>>(image_data: &[u8], file_path: P) -> io::Result<()> {
    let num_rows = 28;
    let num_cols = 28;
    let mut img = GrayImage::new(num_cols, num_rows);

    for (i, pixel) in image_data.iter().enumerate() {
        let x = (i % num_cols as usize) as u32;
        let y = (i / num_cols as usize) as u32;
        img.put_pixel(x, y, Luma([*pixel]));
    }

    img.save(file_path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    Ok(())
}