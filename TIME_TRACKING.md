# Time Tracking Features for InertiaLink Training

This document describes the enhanced time tracking and progress monitoring features added to the training pipeline.

## Features Added

### 1. Dynamic Time Estimation During Training
- **Epoch ETA**: Real-time estimate of time remaining for current epoch
- **Total ETA**: Dynamic estimate of total training time remaining
- **Progress Bar**: Enhanced progress display with time information

### 2. Comprehensive Time Tracking
- **Epoch Duration**: Tracks time taken for each epoch
- **Training Summary**: Complete time analysis at training completion
- **Performance Metrics**: Fastest/slowest epoch, variance analysis

### 3. Enhanced Docker Support
- **Environment Variables**: Optimized for time tracking
- **Progress Monitoring**: Better logging and output buffering
- **Testing Support**: Dedicated test profile for validation

## Usage Examples

### Basic Training with Time Tracking
```bash
# Using Docker Compose
docker-compose up inertialink-xpu

# Using Docker directly
docker run --rm -it \
  --device=/dev/dri \
  --privileged \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  inertialink-xpu:latest
```

### Quick Time Tracking Test
```bash
# Test time tracking functionality
docker-compose --profile test up time-tracking-test

# Or run directly
python3 scripts/test_time_tracking.py 5 20 0.1
```

### Custom Time Tracking Parameters
```bash
# Test with custom parameters
python3 scripts/test_time_tracking.py <epochs> <batches> <duration_per_batch>

# Example: 10 epochs, 50 batches, 0.2s per batch
python3 scripts/test_time_tracking.py 10 50 0.2
```

## Output Examples

### During Training
```
Epoch   25/150  [###############-------]  batch 125/250  lr=3.14e-04  epoch_eta=2.3m  total_eta=45.6m
```

### Epoch Completion
```
Epoch  25/150 completed in 3.2m (192s)
Training time: 78.5m total, avg: 3.1m/epoch, ETA: 387.2m
```

### Final Summary
```
============================================================
TRAINING TIME SUMMARY
============================================================
Total training time: 8.45 hours (506.8 minutes)
Epochs completed: 150
Average epoch time: 3.38 minutes
Fastest epoch: 2.95 minutes
Slowest epoch: 4.12 minutes
Epoch time variance: 1.17 minutes
Training completed at: 2026-04-23 10:27:45
============================================================
```

## Environment Variables

### Time Tracking Configuration
- `PYTHONUNBUFFERED=1`: Ensures real-time output in Docker
- `TZ=UTC`: Consistent timezone for time tracking
- `TRAIN_GPU_FAST=1`: Enables fast mode for better time estimates

### Training Configuration
- `TRAIN_EPOCHS=150`: Number of training epochs
- `TRAIN_BATCH_SIZE=128`: Batch size for training
- `TRAIN_VAL_EVERY=3`: Validation frequency
- `TRAIN_PROGRESS_EVERY=25`: Progress update frequency

## Docker Enhancements

### New Dependencies
- `tqdm`: Progress bar support
- Enhanced environment variables for better output buffering

### Docker Compose Services
- `inertialink-xpu`: Main training service with time tracking
- `time-tracking-test`: Test service for validation

## Implementation Details

### Time Calculation Methods
1. **Epoch ETA**: Based on current batch progress and average batch time
2. **Total ETA**: Based on historical epoch times and remaining epochs
3. **Variance Analysis**: Tracks performance consistency across epochs

### Progress Display
- Real-time updates during batch processing
- Automatic unit conversion (seconds → minutes → hours)
- Clean, non-overlapping output formatting

### Error Handling
- Graceful handling of interrupted training
- Time tracking preserved across checkpoints
- Robust estimation for early training phases

## Troubleshooting

### Time Estimates Inaccurate
- Early epochs may have less accurate estimates (normal behavior)
- Estimates improve after 2+ epochs of historical data
- GPU utilization affects batch processing time

### Docker Output Buffering
- Ensure `PYTHONUNBUFFERED=1` is set
- Use `docker-compose logs -f` for real-time output
- Check container logs if updates seem delayed

### Performance Issues
- Time tracking adds minimal overhead (<0.1%)
- High-frequency updates may impact very fast training
- Adjust `TRAIN_PROGRESS_EVERY` if needed

## Integration Notes

The time tracking features are fully integrated with:
- Existing checkpoint/resume functionality
- Mixed precision training (AMP/BF16)
- Multi-device support (CUDA/XPU/CPU)
- All existing training hyperparameters

No breaking changes were introduced - all existing functionality remains unchanged.
