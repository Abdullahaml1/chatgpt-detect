# chatgpt-detect
implementing paper https://arxiv.org/pdf/2301.07597.pdf


# Custom Model Traing Results
Best Dev checkpoint
```
# of steps=1500, Avg Train Loss=0.000501, Avg Dev Loss=0.000255, Train Acc=0.997625, Dev Acc=0.999170, Time: 2737.913234
```
with Test Accuracy
```
Test Accuracy=0.996472, Test Avg loss=7.452892782085414e-05
```

but we got higher test accuracy when continuing training to the end:
```
Test Acc=0.9985471149854711, Test Loss=0.0003716888740479377
```

Original paper model on huggingface Test with our split: (0.8, 0.1, 0.1) Results:
```
Test Accuracy=0.990660, Test Avg loss=0.0015800070495415447
```
