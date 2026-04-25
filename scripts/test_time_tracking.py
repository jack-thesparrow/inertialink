#!/usr/bin/env python3
"""
test_time_tracking.py — Simple test script to verify time tracking functionality
Tests the time estimation and progress display features added to training script.
"""

import time
import sys
import os

def simulate_training_progress(epochs=5, batches_per_epoch=20, batch_duration=0.1):
    """Simulate training with time tracking similar to the main training script."""
    
    print("=== Time Tracking Test ===")
    print(f"Simulating {epochs} epochs with {batches_per_epoch} batches each")
    print(f"Batch duration: {batch_duration}s")
    print("-" * 60)
    
    training_start_time = time.time()
    epoch_start_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        for batch in range(batches_per_epoch):
            # Simulate batch processing
            time.sleep(batch_duration)
            
            # Progress display with time estimates (similar to main script)
            if batch % 5 == 0 or batch == batches_per_epoch - 1:
                current_time = time.time()
                epoch_elapsed = current_time - epoch_start_time
                training_elapsed = current_time - training_start_time
                
                # Estimate time remaining for current epoch
                if batch > 0:
                    epoch_eta = epoch_elapsed * (batches_per_epoch / (batch + 1) - 1)
                    epoch_eta_str = f"{epoch_eta/60:.1f}m" if epoch_eta > 60 else f"{epoch_eta:.0f}s"
                else:
                    epoch_eta_str = "--s"
                
                # Estimate time remaining for total training
                epochs_completed = epoch
                epochs_remaining = epochs - epoch - 1
                if epochs_completed > 0:
                    avg_epoch_time = training_elapsed / epochs_completed
                    total_eta = avg_epoch_time * epochs_remaining
                    total_eta_str = f"{total_eta/3600:.1f}h" if total_eta > 3600 else f"{total_eta/60:.1f}m"
                else:
                    total_eta_str = "--m"
                
                bar_width = 20
                filled = int(bar_width * (batch + 1) / batches_per_epoch)
                bar = "#" * filled + "-" * (bar_width - filled)
                
                sys.stdout.write(
                    f"\rEpoch {epoch+1:>2}/{epochs}  [{bar}]  "
                    f"batch {batch+1}/{batches_per_epoch}  "
                    f"epoch_eta={epoch_eta_str}  "
                    f"total_eta={total_eta_str}"
                )
                sys.stdout.flush()
        
        # Epoch completion
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_start_times.append(epoch_duration)
        
        epochs_completed = epoch + 1
        training_elapsed = epoch_end_time - training_start_time
        avg_epoch_time = training_elapsed / epochs_completed
        
        if epochs_completed > 1:
            remaining_epochs = epochs - epoch - 1
            est_remaining_time = avg_epoch_time * remaining_epochs
            est_remaining_str = f"{est_remaining_time/60:.1f}m"
        else:
            est_remaining_str = "Calculating..."
        
        print(f"\nEpoch {epoch+1:>2}/{epochs} completed in {epoch_duration:.1f}s")
        print(f"Training time: {training_elapsed:.1f}s total, avg: {avg_epoch_time:.1f}s/epoch, ETA: {est_remaining_str}")
    
    # Final summary
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"\n{'='*60}")
    print("TRAINING TIME SUMMARY")
    print(f"{'='*60}")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.1f} minutes)")
    print(f"Epochs completed: {epochs}")
    
    if epoch_start_times:
        avg_epoch_time = sum(epoch_start_times) / len(epoch_start_times)
        fastest_epoch = min(epoch_start_times)
        slowest_epoch = max(epoch_start_times)
        
        print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        print(f"Fastest epoch: {fastest_epoch:.2f} seconds")
        print(f"Slowest epoch: {slowest_epoch:.2f} seconds")
        print(f"Epoch time variance: {slowest_epoch - fastest_epoch:.2f} seconds")
    
    print(f"{'='*60}")
    print("✅ Time tracking test completed successfully!")

if __name__ == "__main__":
    # Allow command line arguments to customize test
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    batches = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    simulate_training_progress(epochs, batches, duration)
