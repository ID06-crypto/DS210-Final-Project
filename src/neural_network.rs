// This module is the neural network for the project. It uses the ndarray and ndarray-rand crates for matrix operations and random number generation.
// It also uses the rand crate for random number generation

use crate::file_in_and_out;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use ndarray::{Array2};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use rand::prelude::SliceRandom;
use ndarray_rand::rand_distr::Normal;

// Creates a new hashmap full of wrong data for training the neural net. You should use the path to wikilink_graph_cleaned for this
// The keys are the ids of the nodes and the values are the ids of the nodes they point to
// The function takes the path to the file as input and returns a hashmap of the ids and the values
// It works by reading the file line by line and splitting the lines into ids and values
// It then checks if the key already exists in the hashmap and if it doesn't it pushes the value to the vector of values


pub fn false_data (path : &str) -> HashMap<usize, Vec<usize>> {
    let real_ids : HashMap<usize, Vec<usize>> = file_in_and_out::read_ids(path);
    let mut result : HashMap<usize, Vec<usize>> = HashMap::new();
    let mut rng = rand::thread_rng();
    let keys : &Vec<&usize> = &real_ids.keys().collect();

    for key in real_ids.keys() {
	let mut new_vec : Vec<usize> = Vec::new();
        for _ in 0..10 {
            let mut best_id : usize = *keys[rng.gen_range(0..keys.len())];

            // if this is a real link and not a loop back ignore it
            if real_ids.get(&key).unwrap().contains(&best_id) || best_id == *key {
                continue;
            }

            if !new_vec.contains(&best_id) { 
                new_vec.push(best_id)
            }
            result.insert(*key, new_vec.clone());
        
        }
    }

    return result
}

// for the embeddings paramter use the "read_embeddings" function from the file_in_and_out.rs file
// for the pairs paramter  its the read_ids for true data and false_data for the false
// data.  The function takes the embeddings and pairs as input and returns a vector of arrays
// The function works by iterating through the keys of the pairs and and taking the difference of the embeddings
pub fn create_training_data (embeddings : HashMap<usize, Array2<f32>>, pairs : HashMap<usize, Vec<usize>>) -> Vec<Array2<f32>> {
    let mut result : Vec<Array2<f32>> = Vec::new();

    for key in pairs.keys() {
        if embeddings.contains_key(key) {
            let from : Array2<f32> = embeddings.get(key).unwrap().clone();
        
            for val in pairs.get(key).unwrap() {
                if embeddings.contains_key(val) {
                    let to : Array2<f32> = embeddings.get(val).unwrap().clone();
                    let returned : Array2<f32> = from.clone() - &to;
        
                    result.push(returned);
                }

            }
        }
    }

    return result
}

// This struct represents the neural network. It contains the weights and biases of the network and the activations of the layers
// It also contains the learning rate and the output size of the network
pub struct NeuralNetwork {
    layer1_activation: Array2<f32>,
    layer2_activation: Array2<f32>,
    this_guess: Array2<f32>,
    weights_input_to_layer1: Array2<f32>,
    weights_layer1_to_layer2: Array2<f32>,
    weights_layer2_to_output: Array2<f32>,
    bias_layer1 : Array2<f32>,
    bias_layer2 : Array2<f32>,
    bias_output : Array2<f32>,
    output_size: usize,
    learning_rate: f32,
}

impl NeuralNetwork {
    // This function initializes the weights and biases of the network. It takes the input size, layer sizes and output size as input
    // It uses the Xavier uniform initialization method for the weights and initializes the biases to 0. Used for the last layer in 
    // conjunction with the sigmoid activation function
    fn xavier_initialization(input_size: usize, output_size: usize) -> Array2<f32> {
        // Xavier initialization for weights
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        return Array::random((input_size, output_size), Uniform::new(-limit, limit))
    }

    // This function initializes the weights and biases of the network. It takes the input size, layer sizes and output size as input
    // It uses the Kaiming uniform initialization method for the weights and initializes the biases to 0
    // Used for the hidden layers in conjunction with the ReLU activation function
    fn kaiming_initialization(input_size: usize, output_size: usize) -> Array2<f32> {
        // Kaiming initialization for weights
        let dev = (2.0 / input_size as f32).sqrt();
        return Array::random((input_size, output_size), Normal::new(0.0, dev).unwrap())
    }

    // This function creates a new neural network. It takes the input size, layer sizes and output size as input
    // It initializes the weights and biases of the network using the Xavier initialization method
    fn new(input_size: usize, layer1_size: usize, layer2_size:usize, output_size: usize, learning_rate: f32) -> Self {
        // Initialize the weights for the input and hidden layers randomly between 0 and 0.1.
        let weights_input_to_layer1 = Self::kaiming_initialization(input_size, layer1_size);
        let weights_layer1_to_layer2 = Self::kaiming_initialization(layer1_size, layer2_size);
        let weights_layer2_to_output = Self::xavier_initialization(layer2_size, output_size);

        // Initialize the biases for the hidden and output layers to 0
        let bias_layer1 = Array::zeros((1, layer1_size));
        let bias_layer2 = Array::zeros((1, layer2_size));
        let bias_output = Array::zeros((1, output_size));
       

        // Return a neural network that has the randomly initialized weights.
        NeuralNetwork {
            layer1_activation : Array::zeros((1, layer1_size)),
            layer2_activation : Array::zeros((1, layer2_size)),
            this_guess : Array::zeros((1, 1)),
            weights_input_to_layer1,
            weights_layer1_to_layer2,
            weights_layer2_to_output,
            bias_layer1,
            bias_layer2,
            bias_output,
            output_size,
            learning_rate,
        }
    }

    // ReLU activation function and takes as input a matrix and returns a matrix
    // The ReLU function is defined as f(x) = max(0, x)
    fn relu (x : &Array2<f32>) -> Array2<f32> {
        return x.mapv(|x| if x < 0.0 { 0.0 } else { x })
    }

    // Derivative of the ReLU function takes as input a matrix and returns a matrix
    // The derivative of the ReLU function is defined as f'(x) = 1 if x > 0 else 0
    fn relu_derivative (x : &Array2<f32>) -> Array2<f32> {
        return x.mapv(|x| if x < 0.0 { 0.0 } else { 1.0 })
    }


    // Sigmoid activation function takes as input a matrix and returns a matrix
    // The sigmoid function is defined as f(x) = 1 / (1 + e^(-x))
    fn sigmoid (x : &Array2<f32>) -> Array2<f32> {
        return 1.0 / (1.0 + (-x).mapv(f32::exp))
    }

    // Derivative of the sigmoid function takes as input a matrix and returns a matrix
    // The derivative of the sigmoid function is defined as f'(x) = f(x) * (1 - f(x))
    fn sigmoid_derivative (x : &Array2<f32>) -> Array2<f32> {
        return x * &(1.0 - x)
    }

    // Forward propagation.  Returns all of the intermediate and final outputs.
    fn forward(&mut self, input: &Array2<f32>, confidence_yes : f32) -> usize 
    {
        // (layer1_output, layer2_output, final_output)
        self.layer1_activation = Self::relu(&(&(input).dot(&self.weights_input_to_layer1) + &self.bias_layer1));
        self.layer2_activation = Self::relu(&(&self.layer1_activation.dot(&self.weights_layer1_to_layer2) + &self.bias_layer2));
        let final_output = Self::sigmoid(&(&self.layer2_activation.dot(&self.weights_layer2_to_output) + &self.bias_output));
        self.this_guess = final_output.clone();

        let mut best_guess : usize = 0;

        if final_output[[0, 0]] > confidence_yes {
            best_guess = 1;
        } else {
            best_guess = 0;
        }

        return best_guess;
    }

    // Backpropagation pass through the network. No return values but it is supposed to
    // update all weights and biases.  It accepts the input, intermediate outputs and final outputs
    // as parameters as well as the target values.
    fn backward( &mut self, input: &Array2<f32>, target: usize) {
        let target_array = Array2::from_elem((1, 1), target as f32);
        let final_output = self.this_guess.clone();

        let error : Array2<f32> = target_array - final_output.clone();
        let gradient0 : Array2<f32> = &error * Self::sigmoid_derivative(&final_output);

        self.weights_layer2_to_output += &(self.layer2_activation.t().dot(&gradient0) * self.learning_rate);
        self.bias_output -= &(&gradient0 * self.learning_rate);

        let gradient1 : Array2<f32> = Self::relu_derivative(&self.layer2_activation) * gradient0.dot(&self.weights_layer2_to_output.t());
        self.weights_layer1_to_layer2 += &(self.learning_rate * self.layer1_activation.t().dot(&gradient1));
        self.bias_layer2 -= &(&gradient1 * self.learning_rate);

        let gradient2 : Array2<f32> = Self::sigmoid_derivative(&self.layer1_activation) * gradient1.dot(&self.weights_layer1_to_layer2.t());
        self.weights_input_to_layer1 += &(self.learning_rate * &(input).t().dot(&gradient2));
        self.bias_layer1 -= &(&gradient2 * self.learning_rate);
    }
}

// Gives use accuracy of the neural network on the test data
pub fn try_neural_net(embeddings_path: &str, connections_path: &str) {
    // Read the training data and labels
    let embeddings : HashMap<usize, Array2<f32>> = file_in_and_out::read_embeddings(embeddings_path);
    let false_connections : HashMap<usize, Vec<usize>> = false_data(connections_path);
    let true_connections : HashMap<usize, Vec<usize>> = file_in_and_out::read_ids(connections_path);
    let mut true_data : Vec<Array2<f32>> = create_training_data(embeddings.clone(), true_connections);
    let mut false_data : Vec<Array2<f32>> = create_training_data(embeddings.clone(), false_connections.clone());

    true_data.shuffle(&mut rand::thread_rng());
    false_data.shuffle(&mut rand::thread_rng());

    let false_label : usize = 0;
    let true_label : usize = 1;
    let training_true : Vec<Array2<f32>> = true_data[0..true_data.len()/2].to_vec();
    let training_false : Vec<Array2<f32>> = false_data.clone()[0..false_data.len()/2].to_vec();
    
    let testing_true : Vec<Array2<f32>> = true_data[true_data.len()/2..true_data.len()].to_vec();
    let testing_false : Vec<Array2<f32>> = false_data[false_data.len()/2..false_data.len()].to_vec();

    // Initialize the neural network
    let mut nn = NeuralNetwork::new(100, 70, 50, 1, 0.06);

    // Train the neural network 3 times for each data point on both true and false data
    for _iter in 0..3 {
        for sample in training_true.iter() {
            nn.forward(&sample, 0.5);
            nn.backward(&sample, true_label);
        }
        for sample in training_false.iter() {
            nn.forward(&sample, 0.5);
            nn.backward(&sample, false_label);
        }
    }
    
    let mut correct = 0;
    let mut total = 0;

    // Test the neural network
    for sample in testing_true.iter() {
        let output : usize = nn.forward(&sample, 0.5);

        if output == true_label {
            correct += 1;
        }

        total += 1;
    }

    for sample in testing_false.iter() {
        let output : usize = nn.forward(&sample, 0.5);

        if output == false_label {
            correct += 1;
        }

        total += 1;
    }


    println!("Neural Net Accuracy: {}%", (correct as f32 / total as f32) * 100.0);
}