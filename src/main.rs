mod idx_reader;
fn main() -> std::io::Result<()> {
    let image_file_path = "images.idx3-ubyte";
    let label_file_path = "labels.idx1-ubyte";
    let output_dir = "output_images";

    idx_reader::process_idx_files(image_file_path, label_file_path, output_dir)?;

    Ok(())
}
