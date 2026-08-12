def train(model, loader, criterion, optimizer, epochs=150):
    history = {"loss": [], "acc": []}
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)                # forward pass
            loss = criterion(preds, yb)      # BCE loss
            loss.backward()                  # backpropagation
            optimizer.step()                 # weight update
            epoch_loss += loss.item() * xb.size(0)
            correct += ((preds > 0.5).float() == yb).sum().item()
            total += yb.size(0)
        history["loss"].append(epoch_loss / total)
        history["acc"].append(correct / total)
        if epoch % 15 == 0:
            print(f"Epoch {epoch:3d} | Loss: {history['loss'][-1]:.4f} | Acc: {history['acc'][-1]:.4f}")
    return history

history = train(model, train_loader, criterion, optimizer)
