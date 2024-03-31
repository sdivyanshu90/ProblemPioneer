# ML4SCI Task Solutions Repository[CMS]

Welcome! This repository contains solutions to three tasks provided by ML4SCI (Machine Learning for Science). Let's take a look at each task:

## Task Details

1. **Electron/Photon Classification:**
   - Classifying electrons and photons in particle physics experiments using machine learning algorithms. We'll develop models to accurately distinguish between these two particle types based on their characteristics.
   
   **Solution:**
   - [ResNet-15-like architecture](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%201%20-%20Solution/Task%201%20-%20Resnet-15%20like%20Architecture%20-%20Pytorch.ipynb) ([PDF](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%201%20-%20Solution/Task%201%20-%20Resnet-15%20like%20Architecture%20-%20Pytorch.pdf))
   - ROC Curve for ResNet-15 Like model: <br>![roc_resnet_15](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%201%20-%20Solution/resnet-15-pytorch-roc-curve.png)

   **Dataset Description:**
   - 32x32 matrices with two channels:
     - Hit energy
     - Time
   - Two types of particles:
     - Electrons
     - Photons
   - These matrices represent the particles hitting the detector.

2. **Deep Learning-based Quark-Gluon Classification:**
   - Classifying quarks and gluons using deep learning techniques. Quarks and gluons are fundamental particles in the Standard Model of particle physics, and accurate classification is crucial for various physics analyses.
   
   **Solution:**
   - [Model 1: Similar to VGG with 12 layers](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%201%20(VGG12)/Common%20Task%20-%202%20Quark-Gluon%20Classification%20M1.ipynb) ([PDF](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%201%20(VGG12)/Common%20Task%202%20-%20Quark-Gluon%20Classification%20M1.pdf))
   - ROC Curve for Model 1:![roc_curve_model_1](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%201%20(VGG12)/vgg_12_model_roc_2.png)
   - [Model 2: Deep Learning](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%202%20(DL)/Common%20Task%202%20Quark-Gluon%20Classification%20M2%20CNN%20TF.ipynb) ([PDF](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%202%20(DL)/Common%20Task%202%20Quark-Gluon%20Classification%20M2%20CNN%20TF.pdf))
   - ROC Curve for Model 2:![roc_curve_model_2](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Common%20Task%202%20-%20Solution/Common%20Task%20-%202%20Model%202%20(DL)/roc_model_2.png)

   **Dataset Description:**
   - 125x125 matrices in three-channel images.
   - Two classes of particles:
     - Quarks
     - Gluons
   - The images depict particles impinging on a calorimeter.

3. **CMS Project Task 3g:**
   - This task focuses on particle momentum estimation in the CMS (Compact Muon Solenoid) project. We'll explore graph neural networks (GNN) for momentum regression in the CMS Trigger System.
   
   **Discussion:**
   - Both GCN (Graph Convolutional Network) and GAT (Graph Attention Network) models, which work with graphs, show good results when compared to traditional methods like CNNs. However, to make these models work well, it's important to create good graphs. This means we need to explore better ways to build graphs to improve the models in the future.
   
   - When it comes to how fast they work, GCN usually works faster than GAT. This is because GCN has a simpler design and works in a more straightforward way. It takes less time to train and doesn't need as much computing power. But even though GCN is faster, it can still perform as well as GAT, which is more complex. This shows that both methods are effective in capturing relationships in graphs.

   **Solution:**
   - [Model 1: Graph Convolutional Network (GCN)](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%201%20-%20GCN/Task%203g%20GCN%20model.ipynb) ([PDF](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%201%20-%20GCN/Task%203g%20GCN%20model.pdf))
   - ROC Curve for GCN Model:![gcn_roc_curve](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%201%20-%20GCN/gcn-roc-curve.png)
   - [Model 2: Graph Attention Network (GAT)](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%202%20-%20GAT/Task%203g%20GAT%20model.ipynb) ([PDF](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%202%20-%20GAT/Task%203g%20GAT%20model.pdf))
   - ROC Curve for GAT Model:<br>
     ![gat_roc_curve](https://github.com/sdivyanshu90/ProblemPioneer/blob/main/Specific%20Task%20-%203g%20Solution/GNN%20Model%20-%202%20-%20GAT/gat-roc-curve.png)

   **Dataset Description:**
   - Identical to that of Common Task 2.
   - 125x125 matrices in three-channel images, depicting particles impinging on a calorimeter.

Thank you for visiting the ML4SCI Task Solutions Repository!
