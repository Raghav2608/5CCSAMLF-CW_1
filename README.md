# 5CCSAMLF-CW_1
Repository for the Machine Learning Coursework Kings College London 


Top features RandomForest

depth: 0.344, 
b3: 0.079, 
b1: 0.059, 
a1: 0.055, 
a4: 0.038, 
a3: 0.031, 
a5: 0.022, 
a2: 0.021, 
b2: 0.021, 
a6: 0.021, 
b10: 0.021, 
b9: 0.021, 
b7: 0.021, 
b6: 0.020, 
b5: 0.020, 
b4: 0.020, 
a7: 0.020, 
a10: 0.020, 
a8: 0.020, 
a9: 0.019, 
b8: 0.019, 
price: 0.016, 
table: 0.010, 
z: 0.008, 
clarity: 0.008,
color: 0.008, 
y: 0.007, 
x: 0.007, 
carat: 0.005, 
cut: 0.004




Pearson Correlation 

High positive values 
b3: 0.226, 
b1: 0.175, 
a1: 0.150, 
a4: 0.125, 
table: 0.117, 
cut: 0.042, 
x: 0.023, 
price: 0.021,
b5: 0.019, 
a2: 0.018, 
a8: 0.017, 
y: 0.015, 
clarity: 0.015, 
a7: 0.014, 
b6: 0.009, 
a3: 0.006, 
carat: 0.005, 
a10: 0.004, 
a6: 0.003, 
b2: 0, 
a9: 0, 

High negative coefficient value
depth: -0.409,
b10: -0.032,
z: -0.029,
b9: -0.026,
b4: -0.002, 
color: -0.024,
a5: -0.007, 
b7: -0.013, 
b8: -0.011, 

 


Best model Hyperparameters 
BaggedGradientBoosting(n=25,sample_fraction=1,seed=42,n_estimators=2000,max_depth=3,max_bin=100,reg_alpha=100,
                      booster="gbtree",tree_method="hist",grow_policy="lossguide",
                      eta=0.007,subsample=0.8)
