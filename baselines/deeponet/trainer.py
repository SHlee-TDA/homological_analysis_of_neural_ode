# trainer.py
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, 
                 model,
                 train_loader, val_loader, 
                 config, 
                 criterion=None
                 ):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        
        # Joint optimizer for both networks
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
        )
        
        self.criterion = criterion if criterion is not None else nn.MSELoss()
         
        # Setup results directory
        self.results_path = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)
        
    def train(self):
        print(f"Training models for {self.config.EXPERIMENT_NAME}...")

        history = defaultdict(list)
        best_val_loss = np.inf
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            
            train_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
            
            for input_batch, target_batch in pbar:
                # Unpack inputs: (u, y)
                # input_batch is a list/tuple: [u_batch, y_batch]
                u_batch = input_batch[0].to(self.config.DEVICE)
                y_batch = input_batch[1].to(self.config.DEVICE)
                target_batch = target_batch.to(self.config.DEVICE)


                # Forward
                preds = self.model(u_batch, y_batch)
                loss = self.criterion(preds, target_batch)                
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # logging
                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = self.evaluate(self.val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0: 
                print(f"Epoch {epoch+1:04d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_checkpoint(epoch, best_val_loss)
        
        self._save_final_model()
        return self.model, history
    
    @torch.no_grad()
    def evaluate(self, loader):
        
        self.model.eval()
        val_losses = []
        
        for input_batch, target_batch in loader:
            u_batch = input_batch[0].to(self.device)
            y_batch = input_batch[1].to(self.device)
            target_batch = target_batch.to(self.device)

            preds = self.model(u_batch, y_batch)
            loss = self.criterion(preds, target_batch)
            val_losses.append(loss.item())
            
        return np.mean(val_losses)
    
    def _save_checkpoint(self, epoch, loss):
        save_path = os.path.join(self.results_path, 'best_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': loss
        }, save_path)

    def _save_final_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.results_path, 'final_model.pth'))