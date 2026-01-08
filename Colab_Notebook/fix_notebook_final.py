
import json
import os

nb_path = r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v20.ipynb"

def fix_notebook_final():
    if not os.path.exists(nb_path):
        print(f"Error: Not found {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    updates = 0
    
    # helper to check if code exists
    has_train_model = False
    for cell in cells:
        if "def train_model" in "".join(cell['source']):
            has_train_model = True
            break
            
    if not has_train_model:
        print("Missing train_model function. Inserting...")
        
        # New cell content
        train_model_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, lr=0.001):\n",
                "    \"\"\"\n",
                "    Train model and evaluate on validation set.\n",
                "    Returns: (best_model_state_dict, best_val_acc)\n",
                "    \"\"\"\n",
                "    criterion = nn.CrossEntropyLoss()\n",
                "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
                "    \n",
                "    # Create DataLoaders\n",
                "    train_dataset = torch.utils.data.TensorDataset(\n",
                "        torch.FloatTensor(X_train), torch.LongTensor(y_train)\n",
                "    )\n",
                "    val_dataset = torch.utils.data.TensorDataset(\n",
                "        torch.FloatTensor(X_val), torch.LongTensor(y_val)\n",
                "    )\n",
                "    \n",
                "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
                "    val_loader = DataLoader(val_dataset, batch_size=batch_size)\n",
                "    \n",
                "    best_val_acc = 0.0\n",
                "    best_model_state = None\n",
                "    \n",
                "    for epoch in range(epochs):\n",
                "        model.train()\n",
                "        for batch_X, batch_y in train_loader:\n",
                "            batch_X, batch_y = batch_X.to(device), batch_y.to(device)\n",
                "            \n",
                "            optimizer.zero_grad()\n",
                "            outputs = model(batch_X)\n",
                "            loss = criterion(outputs, batch_y)\n",
                "            loss.backward()\n",
                "            optimizer.step()\n",
                "        \n",
                "        # Validation\n",
                "        model.eval()\n",
                "        correct = 0\n",
                "        total = 0\n",
                "        with torch.no_grad():\n",
                "            for batch_X, batch_y in val_loader:\n",
                "                batch_X, batch_y = batch_X.to(device), batch_y.to(device)\n",
                "                outputs = model(batch_X)\n",
                "                predicted = outputs.argmax(dim=1)\n",
                "                total += batch_y.size(0)\n",
                "                correct += (predicted == batch_y).sum().item()\n",
                "        \n",
                "        val_acc = correct / total\n",
                "        \n",
                "        if val_acc > best_val_acc:\n",
                "            best_val_acc = val_acc\n",
                "            best_model_state = model.state_dict()\n",
                "            \n",
                "    # Load best model weights\n",
                "    if best_model_state is not None:\n",
                "        model.load_state_dict(best_model_state)\n",
                "        \n",
                "    return model, best_val_acc\n",
                "\n",
                "print(\"✅ Training function defined\")"
            ]
        }
        
        # Find position to insert: Before "8. Hyperparameter Tuning"
        insert_idx = -1
        for i, cell in enumerate(cells):
            if "## 8. Hyperparameter Tuning" in "".join(cell['source']):
                insert_idx = i
                break
        
        if insert_idx != -1:
            cells.insert(insert_idx, train_model_cell)
            print(f"Inserted train_model at index {insert_idx}")
            updates += 1
        else:
            print("Could not find insertion point (Section 8), appending...")
            cells.append(train_model_cell)
            updates += 1

    if updates > 0:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("No updates needed.")

if __name__ == "__main__":
    fix_notebook_final()
