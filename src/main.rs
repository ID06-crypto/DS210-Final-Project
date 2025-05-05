// This module is the Rust program to test the neural network and graph operations

mod graph_operations;
mod file_in_and_out;
mod neural_network;
use rand::Rng;

fn main() {
    // Tests the Neural Network
    neural_network::try_neural_net(r"clean_embeddings.csv", r"wikilink_graph_cleaned.csv");

    // Tests the all the graph operations
    graph_operations::try_graph(r"tiny_graph.csv", r"wikilink_ids.csv", r"wikilink_titles.csv");
}

// this function is used to test the function "most extroverted" in the graph_operations module
// it takes the paths to the files you want it to work on as input
#[test]
fn test_most_extroverted() {
    let mut graph = graph_operations::Graph::new_graph();
    graph.read_to_graph(r"test_graph.csv", r"test_id.csv", r"test_titles.csv");
    let result = graph.most_extroverted();
    assert_eq!(result.1, 4);
}


// this function is used to test the function "most popular" in the graph_operations module
// it takes the paths to the files you want it to work on as input
#[test]
fn test_most_popular() {
    let mut graph = graph_operations::Graph::new_graph();
    graph.read_to_graph(r"test_graph.csv", r"test_id.csv", r"test_titles.csv");
    let result = graph.most_popular();

    assert_eq!(result.1, 3);
}