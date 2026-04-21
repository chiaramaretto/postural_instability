import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_stacked_dr_sae(
    model,
    loader,
    device,
    lr=1e-3,
    min_epochs=5,
    max_epochs=100,
    target_loss=0.005,
    layer_batch_size=32,
):
    model.to(device)
    criterion = nn.MSELoss()

    if hasattr(loader.dataset, "tensors") and len(loader.dataset.tensors) > 0:
        base_input_data = loader.dataset.tensors[0]
    else:
        cached_batches = [batch[0] for batch in loader]
        base_input_data = torch.cat(cached_batches, dim=0)

    for i in range(len(model.enc_layers)):
        print(f"\n--- Training DR-SAE Layer {i+1}/5 ---")

        encoder_layer = model.enc_layers[i]
        decoder_layer = model.dec_layers[i]
        optimizer = optim.Adam(list(encoder_layer.parameters()) + list(decoder_layer.parameters()), lr=lr)

        layer_loader = DataLoader(
            TensorDataset(base_input_data),
            batch_size=layer_batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        epoch = 0
        pbar = tqdm(total=max_epochs, desc=f"Layer {i+1} Progress")

        while True:
            epoch += 1
            model.train()
            running_loss = 0.0

            for batch in layer_loader:
                inputs = batch[0].to(device, non_blocking=(device.type == "cuda"))

                if i > 0:
                    with torch.no_grad():
                        for j in range(i):
                            inputs = model.selu(model.enc_layers[j](inputs))

                optimizer.zero_grad()
                z = model.selu(encoder_layer(inputs))
                reconstructed = decoder_layer(z)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(layer_loader)
            pbar.update(1)
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})

            if epoch >= min_epochs and avg_loss < target_loss:
                pbar.write(f"Layer {i+1}, Epoch: {epoch}, Loss: {avg_loss:.6f}")
                pbar.close()
                break

            if epoch >= max_epochs:
                pbar.write(f"Layer {i+1} ha raggiunto il limite di salvaguardia ({max_epochs} epoche).")
                pbar.close()
                break

    return model


def train_fusion_block(model, loader, device, lr=1e-3, min_epochs=5, max_epochs=100, target_loss=0.005, block_name="Fusion"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n--- Training {block_name} Block ---")
    epoch = 0
    pbar = tqdm(total=max_epochs, desc=f"{block_name} Training")

    while True:
        model.train()
        running_loss = 0.0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device, non_blocking=(device.type == "cuda"))
            else:
                inputs = batch.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        epoch += 1

        pbar.update(1)
        pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})

        if epoch >= min_epochs and avg_loss < target_loss:
            pbar.write(f"{block_name} target raggiunto! Epoche: {epoch}, Loss finale: {avg_loss:.6f}")
            pbar.close()
            break

        if epoch >= max_epochs:
            pbar.write(f"{block_name} ha raggiunto il limite di salvaguardia ({max_epochs} epoche).")
            pbar.close()
            break

    return model


def fine_tune_lff_clinical(
    model_lff,
    loader,
    device,
    alpha=0.2,
    lr=5e-4,
    max_epochs=20,
    min_epochs=8,
    patience=6,
):
    """Clinical-aware fine-tuning with mixed reconstruction + regression loss."""
    model_lff.to(device)
    reg_head = nn.Linear(model_lff.conv4.out_channels, 1).to(device)

    optimizer = optim.AdamW(
        list(model_lff.parameters()) + list(reg_head.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    recon_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()

    best_loss = float("inf")
    stale = 0

    print("\n--- Clinical-aware fine-tuning (LFF) ---")
    for epoch in range(1, max_epochs + 1):
        model_lff.train()
        reg_head.train()

        total = 0.0
        total_recon = 0.0
        total_reg = 0.0
        n_batches = 0

        for x, y, has_target in loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            has_target = has_target.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()

            recon, latent = model_lff(x)
            recon_loss = recon_criterion(recon, x)

            pooled = torch.mean(latent, dim=2)
            pred = reg_head(pooled).squeeze(-1)

            if has_target.any():
                reg_loss = reg_criterion(pred[has_target], y[has_target])
            else:
                reg_loss = recon_loss.new_zeros(())

            loss = recon_loss + alpha * reg_loss
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            total_recon += float(recon_loss.item())
            total_reg += float(reg_loss.item())
            n_batches += 1

        avg_total = total / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)

        print(
            f"Clinical FT Epoch {epoch:02d} | Total: {avg_total:.4f} | "
            f"Recon: {avg_recon:.4f} | Reg: {avg_reg:.4f}"
        )

        if avg_total < best_loss - 1e-4:
            best_loss = avg_total
            stale = 0
        else:
            stale += 1

        if epoch >= min_epochs and stale >= patience:
            print(f"Clinical FT early stop at epoch {epoch:02d}.")
            break

    return model_lff
