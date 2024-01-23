use image::{DynamicImage, GrayImage, Luma};

fn main() {
    let image = image::open("lab2/images/i_spy.png").expect("Failed to load image.");
    let template = image::open("lab2/images/i_spy_template.png").expect("Failed to load template.");
    
    let _ = heatmap(&image, &template).save("lab2/test.png");
}

fn heatmap(image: &DynamicImage, template: &DynamicImage) -> GrayImage {
    let image = image.to_luma8();
    let template = template.to_luma8();

    let max_diff: u32 = template.width() * template.height() * 255;

    let output_width = image.width() - template.width() + 1;
    let output_height = image.height() - template.height() + 1;
    GrayImage::from_fn(output_width, output_height, |out_x, out_y| {
        let total_diff: f32 = template.enumerate_pixels()
            .map(|(template_x, template_y, template_pixel)| {
                let image_x = out_x + template_x;
                let image_y = out_y + template_y;
                template_pixel[0] as f32 - image.get_pixel(image_x, image_y)[0] as f32
            })
            .sum();

        let lum = ((total_diff / max_diff as f32) * 255.0) as u8;
        Luma([lum])
    })
}
