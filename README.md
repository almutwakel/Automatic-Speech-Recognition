# Automatic-Speech-Recognition
Kaggle Speech Recognition Task with automatic phoneme alignment

Top 5% Submission

Final Model architecture:

Feature Extractor (CNN):
  Conv1d(15, 256, kernel_size=3, padding=1),
  BatchNorm1d(256),
  GELU,
  Conv1d(256, 512, kernel_size=3, padding=1),
  BatchNorm1d(512),
  GELU(),
  Conv1d(512, 512, kernel_size=3, padding=1),
  AvgPool1d(kernel_size=3, stride=2)

LSTM:
  LSTM(input_size = 512, hidden_size = 1024, num_layers = 2, 
    dropout=0.5, batch_first=True, bidirectional=True)
    
Classifier:
  Linear(2048, 2048),
  GELU(),
  Dropout(0.3),
  Linear(2048, 1024),
  GELU(),
  Dropout(0.3),
  Linear(1024, 512),
  GELU(),
  Linear(512, OUT_SIZE)

Training Parameters:
  Epochs: 82
  Optimizer: AdamW
  Learning rate: 1e-3

Ablation Experiments:
1. ReduceLRonPlateau vs CosineAnnealingLR
2. BeamSearch width 3-9
3. hidden_size 256-1024
4. dropout 0-0.5
5. stride = 1 vs 2
6. Classifier cylinder shape vs pyramid
