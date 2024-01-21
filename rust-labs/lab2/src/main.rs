use image::{RgbImage, ImageBuffer, GrayImage};

fn main() {
    let image = image::open("images/i_spy.png");
    let template = image::open("images/i_spy_template.png");
    
    // search_image(img_buff);
}

fn search_image(image: GrayImage, template: GrayImage) {
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    // let results = ImageBuffer::from_fn(512, 512, |x, y| {

    // });
}
