def train_contrastive_gnn(model, 
                          data_loader, 
                          optimizer, 
                          contrastive_loss_fn, 
                          create_positive_sample, 
                          generate_negative_sample, 
                          shuffle_edges,
                          early_stopping,
                          epochs=40,
                          device='cpu'):
    """
    Train a GNN model using contrastive learning with positive and negative graph augmentations.
    
    Args:
        model (nn.Module): The GNN model to train.
        data_loader (DataLoader): PyTorch DataLoader with batches of graph data.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        contrastive_loss_fn (callable): Contrastive loss function (e.g., NTXentLoss).
        create_positive_sample (callable): Function to create a positive (slightly perturbed) graph sample.
        generate_negative_sample (callable): Function to create a negative (disrupted) graph sample.
        shuffle_edges (callable): Function to shuffle edges in a graph (used in negative sampling).
        early_stopping (EarlyStopping): Early stopping object for halting training when convergence is reached.
        epochs (int): Number of training epochs. Default is 40.
        device (str): Device for computation ('cpu' or 'cuda').
    
    Returns:
        list: A list containing average epoch losses.
    """
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Original graph embedding
            z1 = model(batch)

            # Create positive sample and forward pass
            positive_batch = create_positive_sample(batch, mask_ratio=0.3)
            positive_batch = positive_batch.to(device)
            z_pos = model(positive_batch)
            
            # Create negative sample: strong perturbation + shuffled edges
            negative_batch = create_positive_sample(batch, mask_ratio=0.5)
            negative_batch = shuffle_edges(negative_batch)
            negative_batch = generate_negative_sample(negative_batch)
            negative_batch = negative_batch.to(device)
            z_neg = model(negative_batch)
            
            # Compute contrastive loss
            loss_pos = contrastive_loss_fn(z1, z_pos)
            loss_neg = contrastive_loss_fn(z1, z_neg)
            loss = loss_pos - loss_neg
            
            epoch_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if early_stopping(avg_epoch_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    return loss_history
