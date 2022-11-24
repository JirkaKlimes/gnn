fn main() {
    let bchars = "0123456789abcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ";
    let string = "3NRSdj";
    let float = ascii2float(string, bchars);

    println!("{}", float);
}

fn ascii2float(string: &str, bchars: &str) -> f32 {
    let base = bchars.chars().count() as u32;

    let mut n: u32 = 0;
    for c in string.chars() {
        n = n * base + bchars.find(c as char).unwrap() as u32;
    }
    f32::from_bits(n)
}
