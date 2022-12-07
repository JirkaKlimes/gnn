
mod gnn;
use gnn::Gnn;

use std::time::Instant;

fn main() {
    let export_path = "C:/Users/Jirka/Documents/Projects/Python/gnn/export.gnn".to_string();

    let mut gnn = Gnn::from_file(export_path);

    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let now = Instant::now();
    let res = gnn.push(&x);
    println!("Time taken: {:?}", now.elapsed());


    println!("{:?}", res);

}

