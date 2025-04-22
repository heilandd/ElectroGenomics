#' CreateMEAOutputs
#'
#' This function processes multi-electrode array (MEA) data and generates structured graph information
#' suitable for downstream network analysis and graph neural network input.
#'
#' @param mea_data A list containing edge information (`$gg`), node features (`$features`), and electrophysiological data (`$Ephys`).
#' @return A named list with four elements:
#'   - edges: DataFrame of node connections (from, to)
#'   - edges_feature: DataFrame of edge features (connection strength, sum of spikes, and Euclidean distance)
#'   - node_feature: DataFrame of normalized node features (centralities and fire rate)
#'   - nodes: Vector of included node IDs
CreateMEAOutputs <- function(mea_data) {
  edge_features <- mea_data$gg
  node_features <- mea_data$features
  
  # Compute fire rate from electrophysiological data
  f_r <- mea_data$Ephys %>% 
    group_by(from) %>% 
    summarise(fire_rate = sum(spikes)) %>% 
    mutate(ID = from) %>% 
    dplyr::select(ID, fire_rate)
  node_features <- node_features %>% left_join(., f_r)
  
  # Filter consistent nodes and edges
  nodes <- unique(c(edge_features$from, edge_features$to))
  inter <- intersect(nodes, node_features$ID)
  edge_features <- edge_features %>% filter(from %in% inter, to %in% inter)
  node_features <- node_features[node_features$ID %in% inter, ]
  
  # Filter nodes with sufficient connections
  nodes_keep <- edge_features %>% 
    group_by(from) %>% 
    summarise(connections = length(from)) %>% 
    filter(connections > 5) %>% 
    pull(from)
  edge_features <- edge_features %>% filter(from %in% nodes_keep)
  
  # Extract node features
  nodes_select <- node_features$ID
  node_features_select <- node_features[, c("closeness", "betweenness", "degree", "centrality", "page.rank", "fire_rate")] %>% as.data.frame()
  node_features_select[is.na(node_features_select)] <- 0
  names(node_features_select)[5] <- "page_rank"
  
  # Extract edge list
  edges <- edge_features[, c("from", "to")]
  
  # Compute Euclidean distance between node coordinates
  euclidean_distance <- function(x1, y1, x2, y2) {
    sqrt((x2 - x1)^2 + (y2 - y1)^2)
  }
  names(edge_features)[4] <- "connection"
  edges_feature <- edge_features[, c("connection", "sum_spikes")]
  edges_feature$distance <- map(1:nrow(edge_features), ~ euclidean_distance(
    edge_features$from_x[.x], edge_features$from_y[.x],
    edge_features$to_x[.x], edge_features$to_y[.x])) %>% unlist()
  
  if (nrow(node_features_select) != length(nodes_select)) stop("Mismatch with nodes and edge features")
  
  out <- list(edges, edges_feature, node_features_select, nodes_select)
  names(out) <- c("edges", "edges_feature", "node_feature", "nodes")
  return(out)
}

#' QC_graphs
#'
#' Perform quality control by removing weakly connected nodes and corresponding edges.
#'
#' @param network_pram A list with `edges`, `edges_feature`, `node_feature`, and `nodes`.
#' @return A cleaned list in the same format but with sparse nodes and edges removed.
QC_graphs <- function(network_pram) {
  node_feature_update <- network_pram$node_feature %>%
    mutate(from = network_pram$nodes) %>%
    left_join(., network_pram$edges %>%
                group_by(from) %>%
                summarise(connections = length(from)), by = "from") %>%
    filter(connections > 5)
  
  nodes_update <- node_feature_update$from
  
  edges_update <- cbind(network_pram$edges, network_pram$edges_feature) %>%
    filter(from %in% nodes_update, to %in% nodes_update)
  
  node_feature_update <- node_feature_update %>% dplyr::select(-from, -connections)
  
  out <- list(edges_update[, c("from", "to")], edges_update[, -c(1:2)], node_feature_update, nodes_update)
  names(out) <- c("edges", "edges_feature", "node_feature", "nodes")
  return(out)
}

#' CreateFullSubgraph
#'
#' Create 1-hop subgraphs for all nodes in a network using Python's NetworkX via `reticulate`.
#'
#' @param network_pram A list containing graph structure: edges, node_feature, and node list.
#' @return A list of subgraph objects with:
#'   - edges: Edge list of the subgraph
#'   - expression: Node features of the subgraph
#'   - CenterNode: Binary indicator if node is center
#'   - neighborhood: Neighborhood table from NetworkX traversal
CreateFullSubgraph <- function(network_pram) {
  nodes <- network_pram$nodes
  nx <- reticulate::import("networkx")
  G <- nx$from_pandas_edgelist(network_pram$edges, "from", "to")
  
  correct <- map(1:length(nodes), ~ any(network_pram$edges$from == nodes[.x])) %>% unlist()
  nodes <- nodes[correct]
  
  edges_subgraph <- map(1:length(nodes), function(i) {
    subnodes <- nx$single_source_shortest_path_length(G, nodes[i], cutoff = 1)
    nn <- subnodes %>% as.data.frame() %>% t() %>% as.data.frame()
    colnames(nn) <- "NN"
    nn$nodes <- names(subnodes %>% unlist())
    
    subgraph <- G$subgraph(subnodes)
    subgraph <- map_dfr(subgraph$edges, ~ data.frame(from = .x[[1]], to = .x[[2]]))
    
    center_node <- ifelse(unlist(subnodes) == 0, 1, 0)
    
    return(list(edges = subgraph, nodes = names(subnodes), center_node = center_node, neighborhood = nn))
  }, .progress = TRUE)
  
  rownames(network_pram$node_feature) <- network_pram$nodes
  exp <- map(edges_subgraph, ~ network_pram$node_feature[.x$nodes, ], .progress = TRUE)
  
  Subgraph <- map(1:length(edges_subgraph), function(i) {
    list(
      edges = edges_subgraph[[i]]$edges,
      expression = exp[[i]],
      CenterNode = edges_subgraph[[i]]$center_node,
      neighborhood = edges_subgraph[[i]]$neighborhood
    )
  }, .progress = TRUE)
  
  return(Subgraph)
}







