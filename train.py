import torch
from torch.optim import Adam
from models.unet import UNet
from utils.diffusion_utils import DiffusionUtils
from utils.data_utils import get_dataloader
from config import DEVICE, EPOCHS,BATCH_SIZE,IMAGE_SIZE,LEARNING_RATE,DATA_DIR,T
import os

def train():
    model = UNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    diffusion = DiffusionUtils(T)
    dataloader = get_dataloader(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            t = torch.randint(0, T, (config.BATCH_SIZE,), device=config.DEVICE).long()
            x = batch[0].to(config.DEVICE)
            
            loss = diffusion.get_loss(model, x, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        
        if epoch % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
            
            samples = diffusion.sample(model, 4, config.DEVICE, config.IMAGE_SIZE)
            for i, sample in enumerate(samples):
                diffusion.show_tensor_image(sample)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(checkpoint, f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()