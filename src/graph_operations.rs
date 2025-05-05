// This module contains any graph operations - that is more of an anlysis of distace
// beteen nodes and the most popular nodes in the graph

use std::collections::HashMap;
use crate::file_in_and_out;
use rand::Rng;


// This struct represents the graph. It contains a hashmap of the keys and the values of the graph
// and a hashmap of the ids and the names of the nodes. The keys are the ids of the nodes and the values are the ids of the nodes they point to
#[derive(Debug, PartialEq)]
pub struct Graph {
    keys_vertex_graph : HashMap<usize, Vec<usize>>,
    id_names : HashMap<usize, String>,
}

impl Graph {

    // This function creates a new graph. It initializes the keys_vertex_graph and id_names to empty hashmaps
    // and returns the graph
    pub fn new_graph() -> Graph {
        Graph {
            keys_vertex_graph: HashMap::new(),
            id_names: HashMap::new(),
        }
    }

    // This function reads the graph from a csv file. The first column is the id of the node and the second column is the id of the node it points to
    pub fn read_to_graph(&mut self, graph_path : &str, ids_path : &str, titles_path : &str) {
        self.keys_vertex_graph = file_in_and_out::read_ids(graph_path);
        self.id_names = file_in_and_out::read_names(titles_path, ids_path);
    }

    // This function returns the node with the most outgoing edges
    // It iterates through the keys of the keys_vertex_graph and finds the one with the most outgoing edges
    // It returns the name of the node and the number of outgoing edges and takes  self as input
    pub fn most_extroverted(&self) -> (&str, usize) {
        let mut best_id : usize = 0;
        let mut best_count : usize  = 0;

        for key in self.keys_vertex_graph.keys() {
            if self.keys_vertex_graph[key].len() > best_count {
                best_count = self.keys_vertex_graph[key].len();
                best_id = *key;
            }
        }

        if self.id_names.contains_key(&best_id) {
            return (&self.id_names[&best_id], best_count);
        }
        else {
            return ("", 0);
        }
    }

    // This function returns the node with the most incoming edges by iterating through the keys of the keys_vertex_graph
    // and counting the number of times each key appears in the values of the keys_vertex_graph and keeping track of the onewith the mode
    // It takes self as input and returns the name of the node and the number of incoming edges
    pub fn most_popular(&self) -> (&str, usize) {
        let mut best_id : usize = 0;
        let mut best_count : usize  = 0;
        for key in self.keys_vertex_graph.keys() {
            let mut count = 0;
            for vector in self.keys_vertex_graph.values() {
                if vector.contains(key) {
                    count += 1;
                }
            }
            if count > best_count {
                best_count = count;
                best_id = *key;
            }
        }

        return (&self.id_names.get(&best_id).unwrap(), best_count);
    }

    // This function returns the distance between two nodes with breadth first search algorithm 
    // but should be called from distance() function which initializes the visited vector and the distance
    // It takes the two nodes, the initial distance, the current layer of nodes, and the already visited nodes as input
    fn distance_recursive(&self, node_1 : usize, node_2 : usize, initial_distance : usize, this_layer : Vec<usize>, already_visited : Vec<usize>) -> usize {
        let mut distance : usize = initial_distance;
        let mut current_layer : Vec<usize> = this_layer;
        let mut visited : Vec<usize> = already_visited;
        let mut next_layer : Vec<usize> = Vec::new();

        if current_layer.contains(&node_2) {
            return distance;
        }

        else {
            for current_node in &current_layer {
                if current_layer.is_empty() {
                    return distance + 1;
                }
                for child in self.keys_vertex_graph[&current_node].iter() {
                    if visited.contains(child) {
                        continue
                    }
                    else if !self.keys_vertex_graph.contains_key(child) {
                        return distance + 1
                    }
                    else if !visited.contains(child) {
                        next_layer.push(*child);
                        visited.push(*child);
                    }
                }
            }

            distance += 1;

            return self.distance_recursive(node_1, node_2, distance, next_layer, visited);
        }
    }

    // This function returns the distance between two nodes. Helpful becuase it initializes the visited vector and the distance
    // It takes the two nodes as input and returns the distance between them
    // It uses the distance_recursive function to calculate the distance
    fn distance(&self, node_1 : usize, node_2 :usize) -> usize {
        let mut initially_visited : Vec<usize> = Vec::new();
        let mut next_layer : Vec<usize> = Vec::new();
        initially_visited.push(node_1);
        let initial_distance : usize = 0;
        let best : usize = usize::MAX;

        for child in self.keys_vertex_graph[&node_1].iter() {
            if child == &node_2 || !self.keys_vertex_graph.contains_key(child) {
                return 1;
            }
            next_layer.push(*child);
        }

        return self.distance_recursive(node_1, node_2, initial_distance, next_layer, initially_visited);
    }

    // This function returns the average distance between any two nodes in the graph
    // It takes the accuracy as input and returns the average distance between any two nodes in the graph
    // It uses the distance function to calculate the distance between two random nodes in the graph
    // via a random walk algorithm
    fn average_distance(&self, accuracy : usize) -> f64 {
        let mut total_distance : f64 = 0.0;
        let mut count : f64  = 0.0;
        let all_nodes : Vec<usize> = self.keys_vertex_graph.keys().cloned().collect();

        for i in 0..accuracy {
            let mut rng = rand::thread_rng();
            let random_node_1 : usize = all_nodes[rng.gen_range(0..all_nodes.len())];
            let random_node_2 : usize = all_nodes[rng.gen_range(0..all_nodes.len())];
 
            if random_node_1 != random_node_2 {
                total_distance += self.distance(random_node_1, random_node_2) as f64;
                count += 1.0;
            }
        }

        return total_distance / count
    }
}

//this function is used to test the graph operations and takes the paths to the files you want it to work on as input
pub fn try_graph(graph_path : &str, ids_path : &str, titles_path : &str) {
    let mut graph : Graph = Graph::new_graph();
    graph.read_to_graph(graph_path, ids_path, titles_path);

    println!("Graph Facts:");
    println!("Most extroverted node: '{}' with {} outgoing edges", graph.most_extroverted().0, graph.most_extroverted().1);
    println!("Most popular node: '{}' with {} incoming edges", graph.most_popular().0, graph.most_popular().1);
    println!("Average distance between any two nodes: {}", graph.average_distance(10));
}