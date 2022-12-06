use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;

mod activations;
use self::activations::get_func;

pub struct Gnn {
    n_inputs: i32,
    n_outputs: i32,
    n_neurons: i32,
    n_parameters: i32,

    hidden_act_fn_name: String,
    output_act_fn_name: String,

    hidden_act_fn: fn(f32) -> f32,
    output_act_fn: fn(f32) -> f32,

    creation_date: String,
    last_training_date: String,
    save_date: String,

    encoding_base: String,

    calculation_order: Vec<(usize, bool)>,

    pub activations: Vec<f32>,
    order: Vec<f32>,
    digraph: Vec<Vec<usize>>,
    biases: Vec<f32>,
    weights: Vec<Vec<f32>>,
}

impl Gnn {
    fn read_lines(path: String) -> Vec<String> {
        let res = fs::read_to_string(path); //.expect("Can't read the file!");

        let data = match res {
            Ok(string) => string,
            Err(error) => match error.kind() {
                ErrorKind::NotFound => panic!("File not found!"),
                _ => panic!("Error reading file: {:?}", error),
            },
        };

        data.lines().map(|s: &str| s.to_string()).collect()
    }

    fn ascii2float(string: &str, bchars: &String) -> f32 {
        let base = bchars.chars().count() as u32;

        let mut n: u32 = 0;
        for c in string.chars() {
            n = n * base + bchars.find(c as char).unwrap() as u32;
        }
        f32::from_bits(n)
    }

    fn get_calc_order(order: &Vec<f32>) -> Vec<(usize, bool)> {
        let mut pairs: Vec<(usize, f32)> = vec![];
        for i in 0..order.len() {
            if order[i] as i32 != -1 {
                pairs.push((i, order[i]))
            }
        }

        pairs.sort_by(|a, b| (&a.1).total_cmp(&b.1));

        let mut calc_order: Vec<(usize, bool)> = vec![];

        for p in pairs {
            let is_output = p.1 == 1.0;
            calc_order.push((p.0, is_output));
        }
        calc_order
    }

    pub fn from_file(path: String) -> Self {
        let lines = Self::read_lines(path);

        let mut parsed_data: HashMap<String, String> = HashMap::new();

        let mut base: String = "".to_string();
        let mut activations: Vec<f32> = vec![];
        let mut order: Vec<f32> = vec![];
        let mut digraph: Vec<Vec<usize>> = vec![];
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![];

        let mut l: usize = 0;

        while l < lines.len() {
            match lines[l].as_str() {
                "[GNN]" | "[DATES]" => {
                    l += 1;
                    while lines[l] != "" {
                        let mut data = lines[l].split_whitespace();
                        parsed_data.insert(
                            data.next().unwrap().to_string(),
                            data.next().unwrap().to_string(),
                        );
                        l += 1;
                    }
                }
                "[METADATA]" => {
                    l += 1;
                    base = lines[l].clone();

                    for _ in 0..parsed_data["NEURONS"].parse().unwrap() {
                        l += 1;
                        let mut neuron: Vec<f32> = vec![];
                        for i in (0..lines[l].len()).step_by(6) {
                            neuron.push(Self::ascii2float(&lines[l][i..i + 6], &base))
                        }
                        activations.push(neuron.remove(0));
                        order.push(neuron.remove(0));

                        let indices: Vec<usize> = neuron.iter().map(|&v| v as usize).collect();
                        digraph.push(indices);
                    }
                    l += 1;

                    for i in (0..lines[l].len()).step_by(6) {
                        biases.push(Self::ascii2float(&lines[l][i..i + 6], &base))
                    }

                    for _ in 0..parsed_data["NEURONS"].parse().unwrap() {
                        l += 1;
                        let mut neuron: Vec<f32> = vec![];
                        for i in (0..lines[l].len()).step_by(6) {
                            neuron.push(Self::ascii2float(&lines[l][i..i + 6], &base))
                        }
                        weights.push(neuron);
                    }
                }
                _ => {}
            }
            l += 1;
        }

        Gnn {
            n_inputs: parsed_data["INPUTS"].parse().unwrap(),
            n_outputs: parsed_data["OUTPUTS"].parse().unwrap(),
            n_neurons: parsed_data["NEURONS"].parse().unwrap(),
            n_parameters: parsed_data["PARAMETERS"].parse().unwrap(),

            hidden_act_fn_name: parsed_data["HIDDEN_ACT_FN"].clone(),
            output_act_fn_name: parsed_data["OUTPUT_ACT_FN"].clone(),

            hidden_act_fn: get_func(parsed_data["HIDDEN_ACT_FN"].clone()),
            output_act_fn: get_func(parsed_data["OUTPUT_ACT_FN"].clone()),

            creation_date: parsed_data["CREATION_DATE"].clone(),
            last_training_date: parsed_data["LAST_TRAINING_DATE"].clone(),
            save_date: parsed_data["SAVE_DATE"].clone(),

            encoding_base: base,

            calculation_order: Self::get_calc_order(&order),

            activations: activations,
            order: order,
            digraph: digraph,
            biases: biases,
            weights: weights,
        }
    }

    pub fn push(&mut self, x: &Vec<f32>) -> Vec<f32> {
        if x.len() != self.n_inputs as usize {
            panic!(
                "Wrong input size! (was {} but should be {})",
                x.len(),
                self.n_inputs
            );
        }

        for i in 0..self.n_inputs{
            self.activations[i as usize] = x[i as usize];
        }

        for (neuron, is_output) in &self.calculation_order {

            let mut sum: f32 = self.biases[*neuron];
            for i in 0..self.digraph[*neuron].len(){
                let weight = self.weights[*neuron][i];
                sum += weight * self.activations[self.digraph[*neuron][i]]
            }
            if *is_output{
                self.activations[*neuron] = (self.output_act_fn)(sum)
            }
            else{
                self.activations[*neuron] = (self.hidden_act_fn)(sum)
            }
        }

        self.activations[self.n_inputs as usize..(self.n_inputs + self.n_outputs) as usize].to_vec()
    }
}
