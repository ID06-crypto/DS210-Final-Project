// This module contains the functions to read the files and parse them into the data structures used in the program
// It also contains the functions to create the training data and the neural network

use std::fs::File;
use std::io::prelude::*;
use std::collections::HashMap;
use ndarray::{Array2};
use ndarray::Array;

// This function reads the ids from the csv file and returns a hashmap of the ids and the values
// The first column is the id and the second column is the value
// It takes the path to the file as input and returns a hashmap of the ids and the values
pub fn read_ids(path: &str) -> HashMap<usize, Vec<usize>> {
    let mut result : HashMap<usize, Vec<usize>> = HashMap::new();
    let file = File::open(path).expect("Could not open file");
    let buf_reader = std::io::BufReader::new(file).lines();
    let mut row : usize = 0;

    for line in buf_reader {
        if row > 0 {

            let line_str = line.expect("Error reading");
            let v : Vec<usize> = line_str.trim().split(',').map(|num| { num.parse().unwrap() }).collect::<Vec<usize>>();
            let key = v[0];
            let val = v[1];
        
            if result.contains_key(&key) {
                result.get_mut(&key).unwrap().push(val);
            } else {
                result.insert(key, vec![val]);
            }
        }

        row += 1;
    }
    return result;
}

// This function reads the names from the csv file and returns a hashmap of the ids and the names
// The first column is the id and the second column is the name
pub fn read_names<'a>(names_path : &str, id_path : &str) -> HashMap<usize, String> {
    let mut result : HashMap<usize, String> = HashMap::new();
    let file1 = File::open(id_path).expect("Could not open file");
    let buf_reader_id = std::io::BufReader::new(file1).lines();
    let file2 = File::open(names_path).expect("Could not open file");
    let buf_reader_names = std::io::BufReader::new(file2).lines();

    let mut id_vector : Vec<usize> = Vec::new();
    let mut name_vector : Vec<String> = Vec::new();
    let mut row_id : usize = 0;
    let mut row_name : usize = 0;

    for line in buf_reader_id {
        if row_id > 0 {
            let line_str = line.expect("Error reading").trim().parse().expect("Error parsing number");
            let v : usize = line_str;
            if ! id_vector.contains(&v) {
                id_vector.push(v);
            } 
        }
        row_id += 1;
    }

    for line in buf_reader_names {
        if row_name > 0 {
            let line_str = line.expect("Error reading");
            let v : &str = line_str.trim();
            if ! name_vector.contains(&v.to_string()) {
                name_vector.push(v.to_string());
            }
        }

        row_name += 1;
    }

    for (id, name) in id_vector.iter().zip(name_vector.iter()) {
        if result.contains_key(id) {
            *result.get_mut(id).unwrap() = name.clone();
        } else {
            result.insert(id.clone(), name.to_string());
        }
    }

    return result;
}

// This function reads the embeddings from the csv file and returns a hashmap of the ids and the embeddings
// The first column is the id and the second column is the embedding
// It takes the path to the file as input and returns a hashmap of the ids and the embeddings
pub fn read_embeddings(path: &str) -> HashMap<usize, Array2<f32>> {
    let mut result : HashMap<usize, Array2<f32>> = HashMap::new();
    let file = File::open(path).expect("Could not open file");
    let buf_reader = std::io::BufReader::new(file).lines();

    let mut row : usize = 0;

    for line in buf_reader {
        if row > 0 {
            let line_str = line.expect("Error reading");
            let v : Vec<&str> = line_str.trim().split(',').collect();
            let key = v[0].parse::<usize>().unwrap();
            let val = v[1].trim_start_matches('[').trim_end_matches(']');
    
            let nums : Vec<f32> = val.split_whitespace().map(|num| { num.parse::<f32>().unwrap() }).collect();
            let array : Array2<f32> = Array::from_shape_vec((1, nums.len()), nums.clone()).unwrap();
            
            if ! result.contains_key(&key) {
                result.insert(key, array);
            } 
        }
        row += 1;
    }
    return result;
}