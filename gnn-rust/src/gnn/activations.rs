
pub fn get_func(name: String) -> fn(f32) -> f32 {
    match name.as_str(){
        "ReLU" => relu,
        "Identity" => identity,
        _ => panic!("Unkown activation function: {}", name)
    }
}


fn identity(z: f32) -> f32 {
    z
}


fn relu(z: f32) -> f32 {
    if z < 0.0 {
        return 0.0;
    }
    z
}
